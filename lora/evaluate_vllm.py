"""vLLM-based evaluation for trained LoRA adapters.

The PEFT adapter saved by the training pipeline is directly compatible with
vLLM's LoRARequest mechanism — no conversion needed.  vLLM reads the standard
adapter_config.json + adapter_model.safetensors files produced by
model.save_pretrained().

Requirements when initialising the vLLM engine:
    - enable_lora=True
    - max_lora_rank >= lora_rank used during training

Example:
    from lora.evaluate_vllm import evaluate_with_vllm
    from lora.config import LoRAConfig

    cfg = LoRAConfig.from_yaml("lora/configs/example_qwen3b.yaml")
    evaluate_with_vllm(cfg)
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .config import LoRAConfig

logger = logging.getLogger(__name__)


def evaluate_with_vllm(
    config: LoRAConfig,
    lora_path: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
) -> dict:
    """Generate responses using vLLM + LoRA and compare to steered references.

    vLLM loads the adapter via LoRARequest, which is the standard way to serve
    LoRA adapters with vLLM.  The adapter directory must contain the files
    written by PEFT's save_pretrained():

        adapter_config.json
        adapter_model.safetensors   (or adapter_model.bin)

    Args:
        config: Experiment config (model_id, lora_rank, n_eval, …)
        lora_path: Path to the PEFT adapter directory; defaults to
                   config.run_dir / "lora_weights"
        gpu_memory_utilization: Fraction of GPU memory for vLLM
        tensor_parallel_size: Number of GPUs

    Returns:
        Evaluation results dict (also saved to config.run_dir/eval_results_vllm.json)
    """
    if lora_path is None:
        lora_path = str(config.run_dir / "lora_weights")

    if not Path(lora_path).exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {lora_path}\n"
            "Run the training stage first: python -m lora --config <cfg> --stage train"
        )

    # ── Reference data ────────────────────────────────────────────────────────
    with open(config.harmful_prompts_file) as f:
        prompts: List[str] = json.load(f)

    with open(config.harmful_responses_file) as f:
        steered_all = json.load(f)

    angle_key = str(config.steering_angle)
    if angle_key not in steered_all:
        available = sorted(int(k) for k in steered_all.keys())
        raise KeyError(
            f"Angle {config.steering_angle}° not in file. Available: {available}"
        )

    steered_refs: List[str] = steered_all[angle_key]

    baseline_path = (
        Path(config.harmful_responses_file).parent / "harmful-en-baseline.json"
    )
    baseline_refs: Optional[List[str]] = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_refs = json.load(f)

    n = min(config.n_eval, len(prompts))
    eval_prompts = prompts[:n]
    eval_steered = steered_refs[:n]
    eval_baseline = (baseline_refs or [None] * n)[:n]

    # ── vLLM engine ───────────────────────────────────────────────────────────
    logger.info(f"Initialising vLLM: {config.model_id}")
    logger.info(f"  enable_lora=True  max_lora_rank={config.lora_rank}")

    llm = LLM(
        model=config.model_id,
        enable_lora=True,
        # max_lora_rank must cover the rank used during training
        max_lora_rank=max(config.lora_rank, 16),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config.eval_max_new_tokens,
    )

    # ── LoRA request ──────────────────────────────────────────────────────────
    # lora_int_id must be a unique positive integer per adapter in a session
    lora_request = LoRARequest(
        lora_name=config.get_run_name(),
        lora_int_id=1,
        lora_path=lora_path,
    )

    chat_messages = [[{"role": "user", "content": p}] for p in eval_prompts]

    logger.info(f"Generating {n} responses (angle target = {config.steering_angle}°) …")
    outputs = llm.chat(
        chat_messages,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    lora_responses = [o.outputs[0].text for o in outputs]

    # ── Results ───────────────────────────────────────────────────────────────
    examples = [
        {
            "prompt": p,
            "lora_response": lr,
            "steered_reference": sr,
            "baseline": br,
        }
        for p, lr, sr, br in zip(
            eval_prompts, lora_responses, eval_steered, eval_baseline
        )
    ]

    results = {
        "run_name": config.get_run_name(),
        "backend": "vllm",
        "n_eval": n,
        "steering_angle": config.steering_angle,
        "lora_path": lora_path,
        "examples": examples,
    }

    eval_path = config.run_dir / "eval_results_vllm.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"vLLM eval results saved → {eval_path}")

    # ── Print samples ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Sample comparisons  [vLLM, angle={config.steering_angle}°]")
    logger.info(f"{'='*60}")
    for i, ex in enumerate(examples[:3]):
        logger.info(f"\n--- Example {i + 1} ---")
        logger.info(f"Prompt:   {ex['prompt'][:100]}…")
        logger.info(f"LoRA:     {ex['lora_response'][:200]}…")
        logger.info(f"Steered:  {ex['steered_reference'][:200]}…")
        if ex["baseline"]:
            logger.info(f"Baseline: {ex['baseline'][:200]}…")
    logger.info(f"{'='*60}\n")

    return results
