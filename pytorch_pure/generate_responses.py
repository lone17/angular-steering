"""Generate responses with angular steering using pure PyTorch.

This script applies angular steering to model outputs during generation.
Supports both full steering (all tokens) and prompt-only steering.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import add_hooks, get_input_data, tokenize_instructions_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_completions(
    model,
    instructions: List[str],
    tokenizer,
    system_prompt: str = None,
    fwd_pre_hooks: list = None,
    fwd_hooks: list = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prompt_only: bool = False,
):
    """Generate completions with optional steering hooks.

    Args:
        model: HuggingFace model for generation
        instructions: List of instruction strings to generate responses for
        tokenizer: HuggingFace tokenizer
        system_prompt: Optional system prompt to prepend to each instruction
        fwd_pre_hooks: List of (module, hook_fn) tuples for forward pre-hooks
        fwd_hooks: List of (module, hook_fn) tuples for forward hooks
        batch_size: Number of samples to process in parallel
        max_new_tokens: Maximum number of tokens to generate per sample
        temperature: Sampling temperature (0 = greedy decoding)
        top_p: Nucleus sampling threshold
        prompt_only: If True, apply hooks only during first forward pass (prompt processing).
                     Cached steered K/V pairs influence generation (matches vLLM enforce_eager=False).

    Returns:
        List of dicts with 'prompt' and 'response' keys
    """
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []
    completions = []
    num_batches = (len(instructions) + batch_size - 1) // batch_size

    # If prompt_only, wrap hooks ONCE before the batch loop
    if prompt_only:
        wrapped_hooks = []
        for module, hook_fn in fwd_hooks:
            wrapped = create_prompt_only_hook(hook_fn)
            wrapped_hooks.append((module, wrapped))
        fwd_hooks = wrapped_hooks

    for i in tqdm(
        range(0, len(instructions), batch_size),
        total=num_batches,
        desc="Generating" + (" (prompt only)" if prompt_only else ""),
    ):
        batch = instructions[i : i + batch_size]
        inputs = tokenize_instructions_fn(batch, tokenizer, system_prompt)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        batch_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Reset prompt-only hooks for each batch
        if prompt_only:
            for _, hook_fn in fwd_hooks:
                if hasattr(hook_fn, "reset"):
                    hook_fn.reset()

        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks,
            module_forward_hooks=fwd_hooks,
        ):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        batch_responses = tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :], skip_special_tokens=True
        )

        for prompt, response in zip(batch_prompts, batch_responses):
            completions.append({"prompt": prompt, "response": response})

    return completions


# =============================================================================
# Angular Steering Implementation
# =============================================================================


def get_angular_steering_output_hook(
    steering_config: dict,
    target_degree: float,
    adaptive_mode: int = 1,
):
    """Create a hook that applies angular steering to layer outputs.

    Args:
        steering_config: Dict with 'first_direction' and 'second_direction' numpy arrays
        target_degree: Rotation angle in degrees (0-360)
        adaptive_mode: Steering application mode:
                       0 = always steer all activations
                       1 = only steer when activation is aligned with first_direction (conditional)

    Returns:
        Hook function that applies angular steering transformation to module outputs
    """
    first_direction = torch.from_numpy(steering_config["first_direction"])
    second_direction = torch.from_numpy(steering_config["second_direction"])

    # Compute rotation
    device = first_direction.device
    theta_rad = torch.tensor(target_degree * torch.pi / 180.0, device=device)

    # Orthonormalize directions
    b1 = first_direction / first_direction.norm()
    b2 = second_direction - (second_direction @ b1) * b1
    b2 = b2 / b2.norm()

    # Projection matrix
    proj_matrix = torch.outer(b1, b1) + torch.outer(b2, b2)

    # Rotation matrix
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)
    rotation_matrix = torch.stack(
        [torch.stack([cos_theta, -sin_theta]), torch.stack([sin_theta, cos_theta])]
    )

    # Steering vector
    unit_vector = torch.tensor([1.0, 0.0], device=device)
    rotated_2d = rotation_matrix @ unit_vector
    steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

    _cache = {}

    def steering_hook(_module, _input, output):
        device = output.device
        dtype = output.dtype
        cache_key = (device, dtype)

        if cache_key not in _cache:
            _cache[cache_key] = (
                proj_matrix.to(device=device, dtype=dtype),
                steering_vector.to(device=device, dtype=dtype),
                first_direction.to(device=device, dtype=dtype),
            )

        proj, steer, first_dir = _cache[cache_key]

        projected = output @ proj
        scale = projected.norm(dim=-1, keepdim=True)

        if adaptive_mode == 0:
            steered = output - projected + scale * steer
            return steered
        elif adaptive_mode == 1:
            proj_to_first = output @ first_dir
            mask = (proj_to_first > 0).unsqueeze(-1)
            steered = output - projected + scale * steer
            return torch.where(mask, steered, output)
        else:
            raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}")

    return steering_hook


def create_prompt_only_hook(base_hook_fn):
    """Wrap a hook to only apply during first forward pass (prompt processing).

    Args:
        base_hook_fn: The original steering hook function to wrap

    Returns:
        Wrapped hook function that only applies steering on first forward pass.
        The returned function has a .reset() method to reset state between batches.

    Notes:
        With KV caching enabled (default):
        - First pass: processes full prompt, steering is applied
        - Later passes: process 1 token each, no steering but cached K/V are reused
    """
    state = {"is_first_pass": True}

    def prompt_only_hook(module, input_tuple, output):
        if state["is_first_pass"]:
            state["is_first_pass"] = False
            return base_hook_fn(module, input_tuple, output)
        else:
            return output

    def reset():
        state["is_first_pass"] = True

    prompt_only_hook.reset = reset
    return prompt_only_hook


# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses with angular steering (pure PyTorch)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./output",
        help="Directory containing steering configs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save generated responses",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "jp"],
        help="Language for datasets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (lower if OOM)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=10,
        help="Rotation angle step (10 for 36 angles, 30 for 12 angles)",
    )
    parser.add_argument(
        "--adaptive-mode",
        type=int,
        default=1,
        help="Adaptive steering mode (0: all, 1: conditional on harmful direction)",
    )
    parser.add_argument(
        "--strategy-filter",
        type=str,
        default=None,
        help="Filter configs by strategy (e.g., 'max_sim', 'max_norm')",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Apply steering only to prompt (not generation). Matches vLLM enforce_eager=False.",
    )

    args = parser.parse_args()

    # Setup paths
    model_name = args.model.split("/")[-1]
    config_path = Path(args.config_dir) / model_name
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
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

    # Get module dict for hook registration
    module_dict = dict(model.named_modules())

    # Load test data
    logger.info(f"Loading test data ({args.language})...")
    _, data_test = get_input_data("harmful", args.language)
    logger.info(f"Loaded {len(data_test)} test samples")

    # Generate baseline (no steering)
    baseline_file = output_path / f"harmful-{args.language}-baseline.json"
    if not baseline_file.exists():
        logger.info(
            f"Generating baseline responses (no steering{', prompt-only mode' if args.prompt_only else ''})..."
        )
        baseline_responses = generate_completions(
            model=model,
            instructions=data_test,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
            prompt_only=False,  # Baseline never uses steering/hooks
        )
        baseline_responses = [item["response"] for item in baseline_responses]

        with open(baseline_file, "w") as f:
            json.dump(baseline_responses, f, indent=4)
        logger.info(f"Saved baseline to {baseline_file}")
    else:
        logger.info(f"Baseline already exists: {baseline_file}")

    # Find all steering configs
    if not config_path.exists():
        logger.error(f"Config directory not found: {config_path}")
        logger.error("Please run extract_directions.py first!")
        return

    steering_configs = list(config_path.glob("steering_config-*.npy"))
    if not steering_configs:
        logger.error(f"No steering configs found in {config_path}")
        logger.error("Please run extract_directions.py first!")
        return

    logger.info(f"Found {len(steering_configs)} steering config(s)")

    # Process each config
    for config_file in steering_configs:
        # Parse filename
        stem = config_file.stem  # e.g., "steering_config-en-max_sim_15_mid-pca_0"
        parts = stem.split("-")

        if len(parts) < 3:
            logger.warning(f"Skipping {config_file}: unexpected filename format")
            continue

        lang_code = parts[1]
        direction_info = parts[2]  # e.g., "max_sim_15_mid"

        # Filter by language
        if lang_code != args.language:
            logger.info(f"Skipping {config_file.name}: language mismatch")
            continue

        # Filter by strategy if specified
        if args.strategy_filter and args.strategy_filter not in direction_info:
            logger.info(f"Skipping {config_file.name}: strategy filter")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {config_file.name}")
        logger.info(f"{'='*60}")

        # Load config
        config = np.load(config_file, allow_pickle=True).item()

        # Generate responses at different angles
        steered_responses = {}
        sweep_start = time.time()

        for degree in range(0, 360, args.angle_step):
            logger.info(f"  Generating at {degree}° rotation...")

            # Setup steering hooks
            output_hooks = [
                (
                    module_dict[module_name],
                    get_angular_steering_output_hook(
                        steering_config=steering_config,
                        target_degree=degree,
                        adaptive_mode=args.adaptive_mode,
                    ),
                )
                for module_name, steering_config in config.items()
            ]

            # Generate
            completions = generate_completions(
                model=model,
                instructions=data_test,
                tokenizer=tokenizer,
                fwd_hooks=output_hooks,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens,
                prompt_only=args.prompt_only,
            )

            # Extract responses
            responses = [item["response"] for item in completions]
            steered_responses[str(degree)] = responses

        sweep_time = time.time() - sweep_start
        num_angles = 360 // args.angle_step
        logger.info(
            f"  360° sweep completed in {sweep_time:.2f}s "
            f"({num_angles} angles, {sweep_time/num_angles:.2f}s/angle)"
        )

        # Save responses
        adaptive_mode_label = (
            "rotated" if args.adaptive_mode == 0 else f"adaptive_{args.adaptive_mode}"
        )
        output_file = (
            output_path
            / f"harmful-{args.language}-{direction_info}-pca_0-{adaptive_mode_label}.json"
        )
        with open(output_file, "w") as f:
            json.dump(steered_responses, f, indent=4)

        logger.info(f"  Saved to: {output_file}")

    logger.info("\n✓ Generation complete!")
    logger.info(f"  Output directory: {output_path}")
    if args.prompt_only:
        logger.info("\nNote: Steering was applied ONLY to prompt processing.")
        logger.info(
            "      Cached steered K/V pairs influence generation (matches vLLM enforce_eager=False)."
        )


if __name__ == "__main__":
    main()
