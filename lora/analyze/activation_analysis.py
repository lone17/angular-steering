"""Activation-space analysis: h_lora − h_base deltas vs angular steering directions."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import LoRAConfig
from ..model import load_base_model, load_lora_model
from .weight_analysis import load_directions

logger = logging.getLogger(__name__)


def _make_capture_hook(cache: dict, key: str):
    """Return a forward hook that appends the layer output to cache[key]."""

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        cache.setdefault(key, []).append(h.detach().cpu().float())

    return hook_fn


def _get_prompts(config: LoRAConfig, n_samples: int) -> list[str]:
    """Load prompts from the config's harmful_prompts_file."""
    from ..data import load_steered_pairs

    prompts, _ = load_steered_pairs(
        prompts_file=config.harmful_prompts_file,
        steered_responses_file=config.harmful_responses_file,
        steering_angle=config.steering_angle,
        n_samples=n_samples,
        filter_refusals=False,  # want all prompts for activation analysis
        pool_all_angles=False,
    )
    if not prompts:
        raise ValueError(
            "No prompts loaded — check config.harmful_prompts_file and config.harmful_responses_file"
        )
    return prompts


def _run_forward_passes(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    hook_layers: list[int],
    module_dict: dict,
) -> dict[int, np.ndarray]:
    """Register post_attention_layernorm hooks, run prompts, return mean per layer.

    Args:
        model: Model (base or LoRA) — already on device.
        tokenizer: Tokenizer.
        prompts: Input prompt strings.
        batch_size: Batch size for forward passes.
        hook_layers: Layer indices to hook.
        module_dict: dict mapping "model.layers.{L}.post_attention_layernorm" → module.

    Returns:
        {layer_idx: mean_activation [hidden_size]}
    """
    cache: dict[str, list] = {}
    hooks = []

    for L in hook_layers:
        key = f"model.layers.{L}.post_attention_layernorm"
        mod = module_dict.get(key)
        if mod is None:
            logger.warning(f"Module {key} not found in model; skipping layer {L}")
            continue
        hooks.append(mod.register_forward_hook(_make_capture_hook(cache, key)))

    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(next(model.parameters()).device)
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Aggregate: mean over (batch × tokens) for each layer
    layer_means: dict[int, np.ndarray] = {}
    for L in hook_layers:
        key = f"model.layers.{L}.post_attention_layernorm"
        if key not in cache:
            continue
        # List of [batch, seq_len, hidden] tensors
        all_acts = torch.cat(cache[key], dim=0)  # [total_batch, seq_len, hidden]
        layer_means[L] = all_acts.mean(dim=(0, 1)).numpy()

    return layer_means


def run_activation_analysis(
    config: LoRAConfig,
    lora_path: str | Path,
    directions_file: str | Path,
    prompts: Optional[list[str]] = None,
    n_samples: int = 20,
    batch_size: int = 4,
) -> list[dict]:
    """Compute per-layer activation deltas (h_lora − h_base) vs steering directions.

    Sequentially loads base model then LoRA model to avoid GPU OOM.

    Args:
        config: LoRA experiment config.
        lora_path: Path to the LoRA weights directory.
        directions_file: Path to the .npy steering directions file.
        prompts: Override prompts list. If None, loaded from config.
        n_samples: Max prompts to use.
        batch_size: Forward-pass batch size.

    Returns:
        List of dicts, one per layer.
    """
    if prompts is None:
        prompts = _get_prompts(config, n_samples)
    prompts = prompts[:n_samples]
    logger.info(f"Activation analysis: {len(prompts)} prompts, batch_size={batch_size}")

    directions = load_directions(directions_file)

    # Determine which layers to analyse (those present in directions)
    hook_layers = sorted(
        set(int(k.split(".")[2]) for k in directions if "post_attention_layernorm" in k)
    )
    logger.info(f"Hooking {len(hook_layers)} layers")

    # ── 1. Base model ──────────────────────────────────────────────────────────
    logger.info("Loading base model …")
    base_model, tokenizer = load_base_model(config, for_training=False)

    base_module_dict = dict(base_model.named_modules())

    logger.info("Running base forward passes …")
    base_means = _run_forward_passes(
        base_model, tokenizer, prompts, batch_size, hook_layers, base_module_dict
    )

    # ── 2. LoRA model (wraps base_model in-place) ─────────────────────────────
    logger.info("Loading LoRA adapter …")
    lora_model = load_lora_model(base_model, str(lora_path), for_inference=True)

    # Access the inner base model's modules through PeftModel
    lora_inner_dict = dict(lora_model.base_model.model.named_modules())

    logger.info("Running LoRA forward passes …")
    lora_means = _run_forward_passes(
        lora_model, tokenizer, prompts, batch_size, hook_layers, lora_inner_dict
    )

    # ── 3. Compute deltas ──────────────────────────────────────────────────────
    rows = []
    for L in sorted(hook_layers):
        if L not in base_means or L not in lora_means:
            logger.warning(f"Layer {L}: missing activations, skipping")
            continue

        delta_h = lora_means[L] - base_means[L]
        delta_norm = float(np.linalg.norm(delta_h))

        out_key = f"model.layers.{L}.post_attention_layernorm"
        cos_sim: Optional[float]
        if out_key in directions:
            fd = directions[out_key]["first_direction"]
            norm_delta = np.linalg.norm(delta_h)
            norm_fd = np.linalg.norm(fd)
            if norm_delta == 0 or norm_fd == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(delta_h, fd) / (norm_delta * norm_fd))
        else:
            cos_sim = None

        rows.append(
            {
                "layer_idx": L,
                "delta_norm": delta_norm,
                "cos_sim_first_direction": cos_sim,
            }
        )

    return rows
