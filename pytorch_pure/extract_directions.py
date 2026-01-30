"""Extract steering directions from model activations.

This script extracts steering directions by computing activation differences
between harmful and harmless instructions.
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    add_hooks,
    get_input_data,
    get_mlp_input_hook,
    get_residual_hook,
    tokenize_instructions_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Activation Extraction
# =============================================================================


def extract_activations(
    model,
    instructions: list[str],
    tokenizer,
    layers: list[int],
    positions: list[str],
    batch_size: int = 8,
):
    """Extract activations from specified layers and positions.

    Args:
        model: HuggingFace PreTrainedModel
        instructions: List of instruction strings
        tokenizer: HuggingFace PreTrainedTokenizer
        layers: Layer indices to extract from (e.g., [10, 15, 20])
        positions: Positions within layers - 'mid' (after attention) and/or 'post' (after MLP)
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping 'layer_{idx}_{position}' keys to activation tensors.
        Activations have shape (num_samples, hidden_dim) with last token extracted.
    """
    # Prepare cache
    cache = {}

    # Get module dict for hook registration
    module_dict = dict(model.named_modules())

    # Setup hooks for each layer and position
    # To match TransformerLens's resid_mid and resid_post:
    # - "mid": residual stream after self-attention (before layernorm) → pre-hook on post_attention_layernorm
    # - "post": residual stream after MLP (end of block) → forward hook on layer output

    forward_hooks = []
    pre_hooks = []
    for layer_idx in layers:
        layer_name = f"model.layers.{layer_idx}"

        if layer_name in module_dict:
            # Hook for resid_post (end of transformer block)
            if "post" in positions:
                cache_key_prefix = f"layer_{layer_idx}"
                forward_hooks.append(
                    (
                        module_dict[layer_name],
                        get_residual_hook(
                            cache_key_prefix, cache, ["post"], extract_positions=[-1]
                        ),
                    )
                )

            # Hook for resid_mid (after attention, before post_attention_layernorm)
            if "mid" in positions:
                layernorm_name = f"{layer_name}.post_attention_layernorm"
                if layernorm_name in module_dict:
                    cache_key = f"layer_{layer_idx}_mid"
                    pre_hooks.append(
                        (
                            module_dict[layernorm_name],
                            get_mlp_input_hook(
                                cache_key, cache, extract_positions=[-1]
                            ),
                        )
                    )

    # Tokenize ALL instructions at once (matches angular_steering.ipynb behavior)
    # This ensures consistent padding across all batches
    logger.info(f"Tokenizing {len(instructions)} instructions...")
    tokenized = tokenize_instructions_fn(instructions, tokenizer)
    logger.info(f"  Tokenized shape: {tokenized.input_ids.shape}")

    all_input_ids = tokenized.input_ids
    all_attention_mask = tokenized.attention_mask

    # Run forward passes with hooks in batches
    logger.info(f"Extracting activations from {len(instructions)} samples...")
    with add_hooks(
        module_forward_pre_hooks=pre_hooks, module_forward_hooks=forward_hooks
    ):
        with torch.no_grad():
            for i in tqdm(
                range(0, len(instructions), batch_size),
                total=(len(instructions) + batch_size - 1) // batch_size,
                desc="Forward passes",
            ):
                batch_input_ids = all_input_ids[i : i + batch_size]
                batch_attention_mask = all_attention_mask[i : i + batch_size]

                _ = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                )

    # Organize activations
    # Convert cache to (layer, position, batch, hidden_dim) format
    activations = {}
    for key, value in cache.items():
        # key format: "layer_{idx}_{position}"
        activations[key] = value.squeeze(
            1
        )  # Remove token dim (we only kept last token)

    return activations


def compute_steering_directions(
    harmful_acts: dict, harmless_acts: dict, strategy: str = "both"
):
    """Compute steering directions from activations.

    Args:
        harmful_acts: Activations for harmful instructions, keyed by 'layer_{idx}_{position}'
        harmless_acts: Activations for harmless instructions, keyed by 'layer_{idx}_{position}'
        strategy: Layer selection strategy - 'max_sim', 'max_norm', or 'both'

    Returns:
        Dictionary mapping strategy name to steering config dict.
        Each config contains: {'layer': int, 'position': str, 'first_direction': array, 'second_direction': array}
    """
    # Compute candidate directions for all layers/positions
    candidate_directions = {}
    norms = {}

    for key in harmful_acts.keys():
        harmful = harmful_acts[key].float()  # (batch, hidden_dim) - convert to float32
        harmless = harmless_acts[
            key
        ].float()  # (batch, hidden_dim) - convert to float32

        # Normalize each activation sample first (per-sample normalization)
        # This matches the parent implementation: harmful_acts / harmful_acts.norm(dim=-1, keepdim=True)
        harmful_normed = harmful / harmful.norm(dim=-1, keepdim=True)
        harmless_normed = harmless / harmless.norm(dim=-1, keepdim=True)

        # Compute mean of normalized activations
        harmful_mean = harmful_normed.mean(dim=0)
        harmless_mean = harmless_normed.mean(dim=0)

        # Normalize means again
        harmful_mean_norm = harmful_mean / harmful_mean.norm()
        harmless_mean_norm = harmless_mean / harmless_mean.norm()

        # Candidate direction (normalized difference)
        diff = harmful_mean_norm - harmless_mean_norm
        candidate_directions[key] = diff
        norms[key] = diff.norm()

    # Define numeric sort function for consistent layer ordering
    def sort_key(k):
        """Sort keys like 'layer_25_mid' by (layer_idx, position_idx)."""
        parts = k.split("_")
        layer_idx = int(parts[1])
        position = parts[2]
        pos_idx = 0 if position == "mid" else 1
        return (layer_idx, pos_idx)

    # Stack all candidate directions for PCA
    all_candidates = torch.stack(
        [
            candidate_directions[key]
            for key in sorted(candidate_directions.keys(), key=sort_key)
        ]
    )

    # Get device from the first candidate
    device = all_candidates.device

    # Fit PCA on all candidate directions (already in float32)
    pca = PCA()
    pca.fit(all_candidates.cpu().numpy())
    second_direction_pca = torch.from_numpy(pca.components_[0]).to(device)

    # Select layer based on strategy
    directions = {}

    if strategy in ["max_sim", "both"]:
        # Max similarity: highest mean pairwise cosine similarity
        # Normalize all candidates
        candidates_normalized = {
            k: v / v.norm() for k, v in candidate_directions.items()
        }
        candidates_stack = torch.stack(
            [
                candidates_normalized[key]
                for key in sorted(candidates_normalized.keys(), key=sort_key)
            ]
        )

        # Compute pairwise cosine similarities
        pairwise_cosine = candidates_stack @ candidates_stack.T
        mean_cosine = pairwise_cosine.mean(dim=-1)

        # Find layer with highest mean cosine similarity
        max_idx = mean_cosine.argmax().item()
        selected_key = sorted(candidate_directions.keys(), key=sort_key)[max_idx]

        # Log layer selection info
        logger.info(f"\n  Max sim layer selection:")
        for i, key in enumerate(sorted(candidate_directions.keys(), key=sort_key)):
            layer_num = int(key.split("_")[1])
            marker = " ← SELECTED" if i == max_idx else ""
            logger.info(
                f"    Layer {layer_num}: cosine={mean_cosine[i].item():.4f}{marker}"
            )

        # Parse layer and position from key
        parts = selected_key.split("_")
        layer_idx = int(parts[1])
        position = parts[2]

        first_direction = candidate_directions[selected_key]
        first_direction = first_direction / first_direction.norm()

        # DO NOT orthogonalize second direction here - match parent behavior
        # Parent saves PCA component directly without orthogonalization
        # Orthogonalization happens at runtime in _get_rotation_args
        second_direction = second_direction_pca

        directions["max_sim"] = {
            "layer": layer_idx,
            "position": position,
            "first_direction": first_direction.cpu().numpy(),
            "second_direction": second_direction.cpu().numpy(),
        }

    if strategy in ["max_norm", "both"]:
        # Max norm: highest norm of candidate direction
        max_key = max(norms.keys(), key=lambda k: norms[k])

        # Parse layer and position from key
        parts = max_key.split("_")
        layer_idx = int(parts[1])
        position = parts[2]

        first_direction = candidate_directions[max_key]
        first_direction = first_direction / first_direction.norm()

        # DO NOT orthogonalize second direction here - match parent behavior
        # Parent saves PCA component directly without orthogonalization
        # Orthogonalization happens at runtime in _get_rotation_args
        second_direction = second_direction_pca

        directions["max_norm"] = {
            "layer": layer_idx,
            "position": position,
            "first_direction": first_direction.cpu().numpy(),
            "second_direction": second_direction.cpu().numpy(),
        }

    return directions


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering directions from model activations"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save steering configs",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "jp"],
        help="Language for datasets",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=512,
        help="Number of samples to use for extraction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forward passes",
    )

    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        default=["mid", "post"],
        choices=["mid", "post"],
        help="Positions to extract: mid (after attention) and/or post (after MLP)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["max_sim", "max_norm", "both"],
        help="Direction computation strategy",
    )

    args = parser.parse_args()

    # Create output directory
    model_name = args.model.split("/")[-1]
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract from all layers
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))

    logger.info(f"Extracting from all {num_layers} layers")
    logger.info(f"Positions: {args.positions}")

    # Load data
    logger.info(f"\nLoading {args.language} datasets...")
    harmful_train, _ = get_input_data("harmful", args.language)
    harmless_train, _ = get_input_data("harmless", args.language)

    harmful_train = harmful_train[: args.n_samples]
    harmless_train = harmless_train[: args.n_samples]

    logger.info(
        f"Using {len(harmful_train)} harmful and {len(harmless_train)} harmless samples"
    )

    # Extract activations
    logger.info("\nExtracting harmful activations...")
    harmful_acts = extract_activations(
        model, harmful_train, tokenizer, layers, args.positions, args.batch_size
    )
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("\nExtracting harmless activations...")
    harmless_acts = extract_activations(
        model, harmless_train, tokenizer, layers, args.positions, args.batch_size
    )
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Compute directions
    logger.info("\nComputing steering directions...")
    directions = compute_steering_directions(harmful_acts, harmless_acts, args.strategy)

    # Save steering configs for ALL layers
    logger.info(f"\nSaving steering configs to {output_path}")
    for strategy, config in directions.items():
        best_layer_idx = config["layer"]
        position = config["position"]
        first_direction = config["first_direction"]
        second_direction = config["second_direction"]

        # Create config dict for ALL layers using the selected strategy's directions
        # Match parent structure: save entries for BOTH layernorm modules
        config_all_layers = {}
        layernorm_modules = ["input_layernorm", "post_attention_layernorm"]

        num_layers = len(layers)
        for layer_idx in layers:
            for module in layernorm_modules:
                if module != "input_layernorm":
                    # post_attention_layernorm: use same layer
                    module_name = f"model.layers.{layer_idx}.{module}"
                elif layer_idx < num_layers - 1:
                    # input_layernorm: use NEXT layer (parent's pattern)
                    module_name = f"model.layers.{layer_idx + 1}.{module}"
                else:
                    # Skip last layer's input_layernorm
                    continue

                config_all_layers[module_name] = {
                    "first_direction": first_direction,
                    "second_direction": second_direction,
                }

        filename = f"steering_config-{args.language}-{strategy}_{best_layer_idx}_{position}-pca_0.npy"
        filepath = output_path / filename

        np.save(filepath, config_all_layers, allow_pickle=True)
        logger.info(
            f"  Saved: {filename} (best: layer {best_layer_idx}, {len(config_all_layers)} module entries)"
        )

    logger.info("\n✓ Direction extraction complete!")
    logger.info(f"  Configs saved to: {output_path}")
    logger.info(f"  Total configs: {len(directions)}")


if __name__ == "__main__":
    main()
