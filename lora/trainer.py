"""LoRA training logic — SFT, SFT-combined, and DPO objectives."""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from .config import LoRAConfig
from .data import (
    DPODataCollator,
    DPODataset,
    SFTDataCollator,
    SFTDataset,
    load_dpo_triples,
    load_harmless_pairs,
    load_steered_pairs,
)
from .model import add_lora, load_base_model

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


def _make_training_args(config: LoRAConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(config.run_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        fp16=False,
        logging_dir=str(config.run_dir / "tb_logs"),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        run_name=config.get_run_name(),
        dataloader_pin_memory=False,
    )


def _save_and_finish(model, tokenizer, config: LoRAConfig, metrics: dict) -> str:
    metrics_path = config.run_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Train metrics saved → {metrics_path}")

    lora_path = config.run_dir / "lora_weights"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    logger.info(f"LoRA weights saved → {lora_path}")
    return str(lora_path)


# =============================================================================
# Objective A — SFT (harmful steered pairs only)
# =============================================================================


def _train_sft(config: LoRAConfig) -> str:
    """Standard SFT on (prompt → steered_refusal) pairs."""
    model, tokenizer = load_base_model(config, for_training=True)
    model = add_lora(model, config)

    angle_desc = (
        "all angles" if config.pool_all_angles else f"angle={config.steering_angle}°"
    )
    logger.info(f"[SFT] Loading harmful pairs: {angle_desc}")
    prompts, responses = load_steered_pairs(
        prompts_file=config.harmful_prompts_file,
        steered_responses_file=config.harmful_responses_file,
        steering_angle=config.steering_angle,
        n_samples=config.n_train,
        pool_all_angles=config.pool_all_angles,
    )
    logger.info(f"  {len(prompts)} harmful pairs loaded")

    dataset = SFTDataset(prompts, responses, tokenizer, config.max_seq_length)
    logger.info(f"  Dataset: {len(dataset)} examples ({dataset.skipped} skipped)")
    if len(dataset) == 0:
        raise ValueError(
            "No valid training examples — check prompts_file and steered_responses_file."
        )

    trainer = Trainer(
        model=model,
        args=_make_training_args(config),
        train_dataset=dataset,
        data_collator=SFTDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    logger.info("Starting SFT training …")
    result = trainer.train()
    return _save_and_finish(model, tokenizer, config, result.metrics)


# =============================================================================
# Objective B — SFT-combined (harmful steered + harmless baseline pooled)
# =============================================================================


def _train_sft_combined(config: LoRAConfig) -> str:
    """SFT on harmful-steered pairs pooled with harmless-baseline pairs.

    The model sees both:
        harmful  prompt → steered refusal     (don't comply with harmful)
        harmless prompt → baseline response   (do comply with harmless)

    This prevents the adapter from collapsing to always refusing regardless
    of whether the prompt is actually harmful.
    """
    if not config.harmless_prompts_file or not config.harmless_responses_file:
        raise ValueError(
            "training_objective='sft_combined' requires "
            "harmless_prompts_file and harmless_responses_file."
        )

    model, tokenizer = load_base_model(config, for_training=True)
    model = add_lora(model, config)

    # ── Harmful pairs ─────────────────────────────────────────────────────────
    angle_desc = (
        "all angles" if config.pool_all_angles else f"angle={config.steering_angle}°"
    )
    logger.info(f"[SFT-combined] Loading harmful pairs: {angle_desc}")
    harm_prompts, harm_responses = load_steered_pairs(
        prompts_file=config.harmful_prompts_file,
        steered_responses_file=config.harmful_responses_file,
        steering_angle=config.steering_angle,
        n_samples=config.n_train,
        pool_all_angles=config.pool_all_angles,
    )
    logger.info(f"  {len(harm_prompts)} harmful pairs loaded")

    # ── Harmless pairs ────────────────────────────────────────────────────────
    logger.info(
        f"[SFT-combined] Loading harmless pairs: " f"angle='{config.harmless_angle}'"
    )
    less_prompts, less_responses = load_harmless_pairs(
        prompts_file=config.harmless_prompts_file,
        responses_file=config.harmless_responses_file,
        angle=config.harmless_angle,
        n_samples=config.n_harmless,
    )
    logger.info(f"  {len(less_prompts)} harmless pairs loaded")

    # ── Pool ──────────────────────────────────────────────────────────────────
    all_prompts = harm_prompts + less_prompts
    all_responses = harm_responses + less_responses
    logger.info(f"  Combined dataset: {len(all_prompts)} pairs total")

    dataset = SFTDataset(all_prompts, all_responses, tokenizer, config.max_seq_length)
    logger.info(f"  Dataset: {len(dataset)} examples ({dataset.skipped} skipped)")
    if len(dataset) == 0:
        raise ValueError("No valid training examples after combining datasets.")

    trainer = Trainer(
        model=model,
        args=_make_training_args(config),
        train_dataset=dataset,
        data_collator=SFTDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    logger.info("Starting SFT-combined training …")
    result = trainer.train()
    return _save_and_finish(model, tokenizer, config, result.metrics)


# =============================================================================
# Objective C — DPO (steered refusal preferred, harmful baseline rejected)
# =============================================================================


class _DPOTrainer(Trainer):
    """HuggingFace Trainer subclass implementing the DPO objective.

    Uses the base model (LoRA adapters disabled) as the implicit reference,
    avoiding the need to load a second copy of the model.

    Loss:
        L = -E[ log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                         - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    where y_w = chosen (steered refusal), y_l = rejected (harmful compliance),
    and β controls the KL-divergence penalty from the reference.
    """

    def __init__(self, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_beta = beta

    @staticmethod
    def _get_batch_logps(
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mean log-prob over non-masked response tokens, one value per sequence.

        Uses mean (not sum) to reduce sensitivity to response length differences
        between chosen and rejected.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Shift: logits predict the *next* token
        logits = outputs.logits[:, :-1, :].float()  # [B, T-1, V]
        target = labels[:, 1:].clone()  # [B, T-1]

        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(2, target.clamp(min=0).unsqueeze(2)).squeeze(
            2
        )  # [B, T-1]

        mask = (target != -100).float()
        # Mean over response tokens; clamp denominator to avoid div-by-zero
        return (token_logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        bsz = inputs["chosen_input_ids"].size(0)

        # Concatenate chosen + rejected along batch dim for one forward pass
        input_ids = torch.cat(
            [inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0
        )
        attn_mask = torch.cat(
            [inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0
        )
        labels = torch.cat([inputs["chosen_labels"], inputs["rejected_labels"]], dim=0)

        # ── Policy log-probs (LoRA active) ────────────────────────────────────
        policy_logps = self._get_batch_logps(model, input_ids, attn_mask, labels)
        policy_chosen = policy_logps[:bsz]
        policy_rejected = policy_logps[bsz:]

        # ── Reference log-probs (LoRA disabled = frozen base model) ──────────
        model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps = self._get_batch_logps(model, input_ids, attn_mask, labels)
        model.enable_adapter_layers()

        ref_chosen = ref_logps[:bsz]
        ref_rejected = ref_logps[bsz:]

        # ── DPO loss ──────────────────────────────────────────────────────────
        chosen_logratios = policy_chosen - ref_chosen
        rejected_logratios = policy_rejected - ref_rejected
        loss = -F.logsigmoid(
            self.dpo_beta * (chosen_logratios - rejected_logratios)
        ).mean()

        return loss


def _train_dpo(config: LoRAConfig) -> str:
    """DPO on (prompt, chosen=steered_refusal, rejected=harmful_baseline) triples.

    The base model (adapters disabled) serves as the reference; no second copy
    of the model is loaded.
    """
    if not config.dpo_rejected_file:
        raise ValueError(
            "training_objective='dpo' requires dpo_rejected_file "
            "(path to baseline harmful responses)."
        )

    model, tokenizer = load_base_model(config, for_training=True)
    model = add_lora(model, config)

    logger.info(
        f"[DPO] Loading triples: chosen=angle {config.steering_angle}°  "
        f"rejected={config.dpo_rejected_file}  beta={config.dpo_beta}"
    )
    prompts, chosen, rejected = load_dpo_triples(
        prompts_file=config.harmful_prompts_file,
        chosen_file=config.harmful_responses_file,
        chosen_angle=config.steering_angle,
        rejected_file=config.dpo_rejected_file,
        rejected_angle=config.dpo_rejected_angle,
        n_samples=config.n_train,
    )
    logger.info(f"  {len(prompts)} triples loaded")
    if len(prompts) == 0:
        raise ValueError(
            "No DPO triples loaded — check harmful_prompts_file, harmful_responses_file, "
            "and dpo_rejected_file are aligned."
        )

    dataset = DPODataset(prompts, chosen, rejected, tokenizer, config.max_seq_length)
    logger.info(f"  Dataset: {len(dataset)} examples ({dataset.skipped} skipped)")
    if len(dataset) == 0:
        raise ValueError("No valid DPO examples after tokenisation.")

    trainer = _DPOTrainer(
        beta=config.dpo_beta,
        model=model,
        args=_make_training_args(config),
        train_dataset=dataset,
        data_collator=DPODataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    logger.info("Starting DPO training …")
    result = trainer.train()
    return _save_and_finish(model, tokenizer, config, result.metrics)


# =============================================================================
# Public entry point
# =============================================================================


_OBJECTIVES = {
    "sft": _train_sft,
    "sft_combined": _train_sft_combined,
    "dpo": _train_dpo,
}


def train(config: LoRAConfig) -> str:
    """Dispatch to the training objective specified in config.training_objective.

    Supported objectives:
        "sft"          — SFT on harmful steered pairs only (default)
        "sft_combined" — SFT on harmful + harmless pairs pooled
        "dpo"          — Direct Preference Optimisation

    Args:
        config: Fully-populated LoRAConfig.

    Returns:
        Path to the saved LoRA weights directory.
    """
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = config.run_dir / "config.yaml"
    config.to_yaml(config_save_path)
    logger.info(f"Config saved → {config_save_path}")

    obj = config.training_objective
    if obj not in _OBJECTIVES:
        raise ValueError(
            f"Unknown training_objective '{obj}'. " f"Choose from: {list(_OBJECTIVES)}"
        )

    logger.info(f"Training objective: {obj}")
    return _OBJECTIVES[obj](config)
