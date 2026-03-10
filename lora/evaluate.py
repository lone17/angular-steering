"""Evaluation: generate responses with the trained LoRA and compare to steering."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from .config import LoRAConfig
from .model import load_base_model, load_lora_model

logger = logging.getLogger(__name__)


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 4,
) -> List[str]:
    """Generate assistant responses for a list of user prompts.

    Uses left-padding so all prompts end at position -1 (required for
    batched generation with causal models).

    Args:
        model: A (possibly PEFT-wrapped) causal LM
        tokenizer: Matching tokenizer
        prompts: List of user turn strings
        max_new_tokens: Max tokens to generate per response
        temperature: 0.0 = greedy decoding
        batch_size: Prompts per forward pass

    Returns:
        List of response strings (decoded, without the prompt)
    """
    model.eval()
    responses: List[str] = []

    # Switch to left-padding for generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]

        # Format each prompt with the model's chat template
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(
            gen_ids[:, prompt_len:], skip_special_tokens=True
        )
        responses.extend(decoded)

    tokenizer.padding_side = orig_padding_side
    return responses


def evaluate(
    config: LoRAConfig,
    lora_path: Optional[str] = None,
) -> dict:
    """Generate LoRA responses and compare them against the steered references.

    Saves a JSON file with side-by-side examples to config.run_dir/eval_results.json.

    Args:
        config: Experiment config
        lora_path: Path to LoRA adapter directory; defaults to
                   config.run_dir / "lora_weights"

    Returns:
        Dictionary with evaluation results and examples
    """
    if lora_path is None:
        lora_path = str(config.run_dir / "lora_weights")

    # ── Load reference data ───────────────────────────────────────────────────
    with open(config.harmful_prompts_file) as f:
        prompts: List[str] = json.load(f)

    with open(config.harmful_responses_file) as f:
        steered_all = json.load(f)

    angle_key = str(config.steering_angle)
    steered_refs: List[str] = steered_all[angle_key]

    # Optionally load baseline (no-steering) references for comparison
    baseline_path = (
        Path(config.harmful_responses_file).parent / "harmful-en-baseline.json"
    )
    baseline_refs: Optional[List[str]] = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_refs = json.load(f)
        logger.info(f"Loaded baseline responses from: {baseline_path}")

    n = min(config.n_eval, len(prompts))
    eval_prompts = prompts[:n]
    eval_steered = steered_refs[:n]
    eval_baseline = baseline_refs[:n] if baseline_refs else [None] * n

    logger.info(
        f"Evaluating on {n} prompts  (steering angle = {config.steering_angle}°)"
    )

    # ── Load model + LoRA ─────────────────────────────────────────────────────
    base_model, tokenizer = load_base_model(config, for_training=False)
    model = load_lora_model(base_model, lora_path)

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info("Generating with LoRA model …")
    lora_responses = generate_responses(
        model,
        tokenizer,
        eval_prompts,
        max_new_tokens=config.eval_max_new_tokens,
        batch_size=config.eval_batch_size,
    )

    # ── Package results ────────────────────────────────────────────────────────
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
        "config_summary": config.summary(),
        "n_eval": n,
        "steering_angle": config.steering_angle,
        "examples": examples,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    eval_path = config.run_dir / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved → {eval_path}")

    # ── Print sample comparisons ──────────────────────────────────────────────
    n_show = min(3, n)
    logger.info(f"\n{'='*60}")
    logger.info(f"Sample comparisons  (angle={config.steering_angle}°)")
    logger.info(f"{'='*60}")
    for i, ex in enumerate(examples[:n_show]):
        logger.info(f"\n--- Example {i + 1} ---")
        logger.info(f"Prompt:    {ex['prompt'][:100]}…")
        logger.info(f"LoRA:      {ex['lora_response'][:200]}…")
        logger.info(f"Steered:   {ex['steered_reference'][:200]}…")
        if ex["baseline"]:
            logger.info(f"Baseline:  {ex['baseline'][:200]}…")
    logger.info(f"{'='*60}\n")

    return results
