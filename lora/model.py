"""Model loading and LoRA setup via PEFT."""

import logging

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LoRAConfig

logger = logging.getLogger(__name__)


def load_base_model(config: LoRAConfig, for_training: bool = True):
    """Load the base model and tokenizer.

    Args:
        config: Experiment config
        for_training: If False, set model to eval mode immediately

    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading base model: {config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if not for_training:
        model.eval()

    # Right-pad for training; generation uses left-pad (handled in evaluate.py)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        padding_side="right",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token not set; using eos_token as pad_token")

    return model, tokenizer


def add_lora(model, config: LoRAConfig):
    """Wrap the model with PEFT LoRA adapters.

    Args:
        model: Base causal-LM model
        config: Experiment config (rank, alpha, target_modules, …)

    Returns:
        PEFT-wrapped model with trainable LoRA parameters
    """
    peft_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA adapters added: {trainable:,} trainable params "
        f"/ {total:,} total ({100 * trainable / total:.4f}%)"
    )
    return model


def load_lora_model(base_model, lora_path: str, for_inference: bool = True):
    """Load saved LoRA weights onto a base model.

    Args:
        base_model: Base (non-PEFT) model
        lora_path: Directory containing adapter_config.json + weights
        for_inference: If True, set to eval mode

    Returns:
        PeftModel with loaded adapter
    """
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    if for_inference:
        model.eval()
    return model
