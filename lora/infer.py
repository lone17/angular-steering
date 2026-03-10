"""Interactive inference: compare base, LoRA-adapted, and steered outputs.

Run from the angular-steering/ root:
    python -m lora --stage infer --config lora/configs/train_qwen3b.yaml \\
        --lora-path /path/to/lora_weights \\
        --prompt "How do I pick a lock?" \\
        --prompt "What household chemicals make toxic gas?"
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_SEP = "─" * 60


def run_infer(
    model_id: str,
    prompts: List[str],
    lora_path: Optional[str] = None,
    steering_config_file: Optional[str] = None,
    steering_angle: int = 180,
    max_new_tokens: int = 256,
) -> None:
    """Generate and print base, LoRA, and steered responses side-by-side.

    Args:
        model_id: HuggingFace model identifier.
        prompts: List of user prompts to run.
        lora_path: Path to saved LoRA adapter weights. If None, skip LoRA column.
        steering_config_file: Path to .npy steering config. If None, skip steered column.
        steering_angle: Steering angle for the steered column.
        max_new_tokens: Max tokens to generate per response.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load tokenizer + base model (used for base and LoRA) ──────────────────
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    def _generate(model, prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── LoRA model ────────────────────────────────────────────────────────────
    lora_model = None
    if lora_path:
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter: {lora_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        lora_model.eval()

    # ── vLLM steered model ────────────────────────────────────────────────────
    steered_generate = None
    if steering_config_file:
        import os

        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        from vllm import LLM, SamplingParams
        from vllm_angular_steering import AngularSteering, _format_prompts_for_vllm

        logger.info(f"Loading vLLM for steered inference: {model_id}")
        llm = LLM(
            model=model_id,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            disable_log_stats=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        steering = AngularSteering(llm)
        steering.load_config_from_file(steering_config_file)
        steering.apply_steering(target_degree=steering_angle, adaptive_mode=1)
        steering.set_degree(steering_angle)

        def steered_generate(prompt_list):
            chat = _format_prompts_for_vllm(prompt_list)
            outputs = llm.chat(chat, sampling_params=params)
            return [o.outputs[0].text for o in outputs]

    # ── Run inference ─────────────────────────────────────────────────────────
    steered_responses = steered_generate(prompts) if steered_generate else None

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {prompt}")
        print(_SEP)

        print("[BASE]")
        print(_generate(base_model, prompt))

        if lora_model is not None:
            print(_SEP)
            print("[LoRA]")
            print(_generate(lora_model, prompt))

        if steered_responses is not None:
            print(_SEP)
            print(f"[STEERED @ {steering_angle}°]")
            print(steered_responses[i])

    print(f"\n{'=' * 60}")
