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
    tokenize_instructions_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Activation Extraction
# =============================================================================


def _detect_layernorm_modules(model) -> list[str]:
    """Detect [pre_attn_ln, pre_mlp_ln] from layer 0's children by trying known names.

    Note: named_children() registration order does not match forward execution order
    in some architectures (e.g. Gemma-2), so structural traversal is unreliable.
    Candidate lists cover all currently supported models and are easy to extend.
    """
    children = dict(model.model.layers[0].named_children())

    # Almost universal; extend if a model uses a different name
    pre_attn_candidates = ["input_layernorm"]
    # Gemma-2 uses pre_feedforward_layernorm; Llama/Qwen use post_attention_layernorm
    pre_mlp_candidates = ["pre_feedforward_layernorm", "post_attention_layernorm"]

    def _find(candidates: list[str], label: str) -> str:
        for name in candidates:
            if name in children:
                return name
        raise ValueError(f"Could not detect {label}. Checked: {candidates}")

    return [_find(pre_attn_candidates, "pre-attn layernorm"),
            _find(pre_mlp_candidates, "pre-MLP layernorm")]


def extract_activations(
    model,
    instructions: list[str],
    tokenizer,
    layers: list[int],
    layernorm_modules: list[str],
    batch_size: int = 8,
):
    """Extract activations by hooking the input of specified layernorm modules.

    Args:
        model: HuggingFace PreTrainedModel
        instructions: List of instruction strings
        tokenizer: HuggingFace PreTrainedTokenizer
        layers: Layer indices to extract from
        layernorm_modules: Short module names to hook, e.g.
            ["input_layernorm", "post_attention_layernorm"]  (Llama/Qwen)
            ["input_layernorm", "pre_feedforward_layernorm"] (Gemma-2)
        batch_size: Batch size for processing

    Returns:
        Dict mapping 'layer_{idx}_{module_name}' keys to tensors of shape
        (num_samples, hidden_dim).
    """
    cache = {}
    module_dict = dict(model.named_modules())

    pre_hooks = []
    for layer_idx in layers:
        for module_name in layernorm_modules:
            full_name = f"model.layers.{layer_idx}.{module_name}"
            if full_name in module_dict:
                cache_key = f"layer_{layer_idx}_{module_name}"
                pre_hooks.append(
                    (
                        module_dict[full_name],
                        get_mlp_input_hook(cache_key, cache, extract_positions=[-1]),
                    )
                )

    # Tokenize ALL instructions at once (matches angular_steering.ipynb behavior)
    # This ensures consistent padding across all batches
    logger.info(f"Tokenizing {len(instructions)} instructions...")
    tokenized = tokenize_instructions_fn(instructions, tokenizer)
    logger.info(f"  Tokenized shape: {tokenized.input_ids.shape}")

    all_input_ids = tokenized.input_ids
    all_attention_mask = tokenized.attention_mask

    logger.info(f"Extracting activations from {len(instructions)} samples...")
    with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
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

    activations = {}
    for key, value in cache.items():
        activations[key] = value.squeeze(1)  # (batch, 1, hidden) → (batch, hidden)

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
        """Sort keys like 'layer_5_input_layernorm' by (layer_idx, module_order).

        input_layernorm (pre-attention) sorts before the pre-MLP layernorm.
        """
        _, layer_str, module_name = k.split("_", 2)
        pos_idx = 0 if module_name == "input_layernorm" else 1
        return (int(layer_str), pos_idx)

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

        # Parse layer and module name from key
        _, layer_str, module_name = selected_key.split("_", 2)
        layer_idx = int(layer_str)
        position = module_name

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

        # Parse layer and module name from key
        _, layer_str, module_name = max_key.split("_", 2)
        layer_idx = int(layer_str)
        position = module_name

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
        "--strategy",
        type=str,
        default="both",
        choices=["max_sim", "max_norm", "both"],
        help="Direction computation strategy",
    )
    parser.add_argument(
        "--layernorm-modules",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Short module names to hook, e.g. --layernorm-modules input_layernorm pre_feedforward_layernorm. "
            "If not provided, auto-detected from model structure."
        ),
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

    # Resolve hook points: use override if provided, otherwise auto-detect
    if args.layernorm_modules:
        layernorm_modules = args.layernorm_modules
        logger.info(f"Hook points (override): {layernorm_modules}")
    else:
        layernorm_modules = _detect_layernorm_modules(model)
        logger.info(f"Hook points (auto-detected): {layernorm_modules}")

    logger.info(f"Extracting from all {num_layers} layers")

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
        model, harmful_train, tokenizer, layers, layernorm_modules, args.batch_size
    )
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("\nExtracting harmless activations...")
    harmless_acts = extract_activations(
        model, harmless_train, tokenizer, layers, layernorm_modules, args.batch_size
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

        # Save directions for ALL layers, keyed by full module path
        config_all_layers = {
            f"model.layers.{layer_idx}.{module_name}": {
                "first_direction": first_direction,
                "second_direction": second_direction,
            }
            for layer_idx in layers
            for module_name in layernorm_modules
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
