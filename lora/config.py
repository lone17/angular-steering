"""Experiment configuration for LoRA training and data preparation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PrepareConfig:
    """Configuration for data preparation stage (--stage prepare).

    Generates steered harmful responses and optionally harmless baseline responses
    for use in subsequent training stages. These prepared datasets are provided to
    LoRAConfig via harmful_responses_file, harmless_responses_file, etc.

    Run from the angular-steering/ root:
        python -m lora --config lora/configs/prepare_qwen3b.yaml --stage prepare
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    # ── Steering config ────────────────────────────────────────────────────────
    # Path to steering config (.npy file) from pytorch_pure/generate_steering_config.py
    steering_config_file: str = ""

    # ── Output ────────────────────────────────────────────────────────────────
    # Base directory for generated data (relative or absolute path).
    # If relative, resolves from current working directory.
    # Defaults to output/{model_short_name}/ if empty.
    data_dir: str = ""

    # ── Data generation options ─────────────────────────────────────────────────
    # Which dataset split to generate from (default: "train")
    data_split: str = "train"

    # Steering angles to generate responses at (empty list = baseline only).
    # Example: [90, 180] generates responses at 90° and 180°.
    # For DPO chosen: typically [180].
    # For SFT-combined: typically [] (baseline only, harmful handled separately).
    data_angles: List[int] = field(default_factory=list)

    # Max harmful prompts to process (-1 = use all available)
    data_n_harmful: int = -1

    # Max harmless prompts to process (-1 = use all available, 0 = skip harmless)
    data_n_harmless: int = -1

    # Adaptive steering mode (1 = enabled, 0 = disabled)
    data_adaptive_mode: int = 1

    # If True, drop (prompt, response) pairs where the primary angle's response
    # contains a refusal marker — keeps only compliance responses.
    # Filtering is applied before saving, so the saved files reflect what will
    # actually be used for training.
    filter_refusals: bool = False

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def resolved_data_dir(self) -> Path:
        """Return the resolved output directory for generated data."""
        if self.data_dir:
            p = Path(self.data_dir)
            # Absolute path: use as-is
            if p.is_absolute():
                return p
            # Relative path: resolve from current working directory
            return Path.cwd() / p
        # Default: output/{model_short_name}/
        model_short = self.model_id.split("/")[-1]
        return Path.cwd() / "output" / model_short

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path) -> "PrepareConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "PrepareConfig":
        return cls(**data)

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable config summary for logging."""
        w = 20
        lines = [
            f"{'model_id':{w}}: {self.model_id}",
            f"{'steering_config':{w}}: {self.steering_config_file}",
            f"{'data_dir':{w}}: {self.resolved_data_dir}",
            f"{'data_split':{w}}: {self.data_split}",
            f"{'data_angles':{w}}: {self.data_angles}",
            f"{'data_n_harmful':{w}}: {'all' if self.data_n_harmful < 0 else self.data_n_harmful}",
            f"{'data_n_harmless':{w}}: {'all' if self.data_n_harmless < 0 else ('skip' if self.data_n_harmless == 0 else self.data_n_harmless)}",
            f"{'data_adaptive_mode':{w}}: {self.data_adaptive_mode}",
            f"{'filter_refusals':{w}}: {self.filter_refusals}",
        ]
        return "\n".join(lines)


@dataclass
class LoRAConfig:
    """Configuration for a LoRA experiment that approximates angular steering.

    The run_name is auto-generated from key parameters so every output directory
    and log file directly reflects the experiment setup:

        {model}__rank{r}__angle{a}__mods-{modules}__data-{dataset_stem}

    Example:
        Qwen2.5-3B-Instruct__rank4__angle180__mods-q+v__data-harmful-en-dir_max_sim_25_mid-pca_0-adaptive_1

    Override with `run_name` in the YAML to use a custom identifier.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    # ── Data ─────────────────────────────────────────────────────────────────
    # JSON list of harmful prompt strings (e.g. harmful-en-samples.json)
    harmful_prompts_file: str = ""

    # JSON dict {angle_str: [response_0, …]} of steered responses,
    # output of vllm_angular_steering or pytorch_pure/generate_responses.py
    harmful_responses_file: str = ""

    # Which angle to select from the steered responses file as the training target.
    steering_angle: int = 180

    # Max training samples (-1 = use all available)
    n_train: int = -1

    # If True, pool responses from every angle in harmful_responses_file rather
    # than only steering_angle.  steering_angle is ignored when this is True.
    pool_all_angles: bool = False

    # ── Training objective ────────────────────────────────────────────────────
    # "sft"          – SFT on steered-refusal pairs only (default)
    # "sft_combined" – SFT on steered-refusal pairs + harmless-baseline pairs
    #                  pooled together; prevents the adapter collapsing to always
    #                  refusing regardless of prompt content.
    # "dpo"          – Direct Preference Optimisation: steered refusal preferred,
    #                  harmful baseline rejected; uses the base model (LoRA
    #                  adapters disabled) as the implicit reference model.
    training_objective: str = "sft"

    # ── sft_combined: harmless baseline data ──────────────────────────────────
    # JSON list of harmless prompt strings.
    harmless_prompts_file: str = ""
    # JSON dict {key: [response, ...]} of unsteered harmless responses.
    harmless_responses_file: str = ""
    # Key to read from harmless_responses_file (default matches prepare stage output).
    harmless_angle: str = "baseline"
    # Max harmless samples to use (-1 = all).
    n_harmless: int = -1

    # ── dpo: rejected (harmful baseline) responses ────────────────────────────
    # Flat JSON list  OR  {angle_str: [response, ...]} dict.
    # Typically the model's unsteered outputs on the harmful prompts.
    dpo_rejected_file: str = ""
    # Which angle key to read when dpo_rejected_file is a dict (None = first key).
    dpo_rejected_angle: Optional[int] = None
    # KL-regularisation coefficient β in the DPO loss.
    dpo_beta: float = 0.1

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # Which linear sub-modules to attach LoRA adapters to.
    # Common choices:
    #   ["q_proj", "v_proj"]          – attention only (light)
    #   ["q_proj", "k_proj", "v_proj", "o_proj"]   – full attention
    #   ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]  – attn + MLP
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    # Base output dir (relative to the lora/ package directory)
    output_dir: str = "output"

    # Custom run identifier; if None, auto-generated from key params
    run_name: Optional[str] = None

    # ── Evaluation ────────────────────────────────────────────────────────────
    n_eval: int = 20
    eval_max_new_tokens: int = 256
    eval_batch_size: int = 4

    # ─────────────────────────────────────────────────────────────────────────

    def get_run_name(self) -> str:
        """Return a traceable run identifier encoding key experiment parameters."""
        if self.run_name:
            return self.run_name

        model_short = self.model_id.split("/")[-1]

        # Short module string: "q_proj,v_proj" → "q+v"
        modules = "+".join(m.replace("_proj", "") for m in self.lora_target_modules)

        # Dataset identifier from filename stem
        dataset_id = (
            Path(self.harmful_responses_file).stem
            if self.harmful_responses_file
            else "unknown"
        )

        obj_tag = (
            ""
            if self.training_objective == "sft"
            else f"__obj-{self.training_objective}"
        )

        return (
            f"{model_short}"
            f"__rank{self.lora_rank}"
            f"__angle{self.steering_angle}"
            f"__mods-{modules}"
            f"__data-{dataset_id}"
            f"{obj_tag}"
        )

    @property
    def run_dir(self) -> Path:
        """Output directory for this run (created on demand)."""
        base = Path(self.output_dir)
        if not base.is_absolute():
            base = Path(__file__).parent / base
        return base / self.get_run_name()

    @property
    def log_path(self) -> Path:
        """Per-run log file path."""
        lora_root = Path(__file__).parent
        return lora_root / "logs" / f"{self.get_run_name()}.log"

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path) -> "LoRAConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        return cls(**data)

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable config summary for logging."""
        w = 22
        lines = [
            f"{'model_id':{w}}: {self.model_id}",
            f"{'run_name':{w}}: {self.get_run_name()}",
            f"{'training_objective':{w}}: {self.training_objective}",
            f"{'lora_rank':{w}}: {self.lora_rank}",
            f"{'lora_alpha':{w}}: {self.lora_alpha}",
            f"{'lora_target_modules':{w}}: {self.lora_target_modules}",
            f"{'steering_angle':{w}}: {self.steering_angle}°",
            f"{'n_train':{w}}: {'all' if self.n_train < 0 else self.n_train}",
            f"{'filter_refusals':{w}}: {self.filter_refusals}",
            f"{'pool_all_angles':{w}}: {self.pool_all_angles}",
            f"{'harmful_prompts_file':{w}}: {self.harmful_prompts_file}",
            f"{'harmful_responses_file':{w}}: {self.harmful_responses_file}",
            f"{'harmless_prompts_file':{w}}: {self.harmless_prompts_file or '—'}",
            f"{'harmless_responses_file':{w}}: {self.harmless_responses_file or '—'}",
            f"{'dpo_rejected_file':{w}}: {self.dpo_rejected_file or '—'}",
            f"{'dpo_beta':{w}}: {self.dpo_beta}",
            f"{'learning_rate':{w}}: {self.learning_rate}",
            f"{'num_epochs':{w}}: {self.num_epochs}",
            f"{'per_device_batch_size':{w}}: {self.per_device_batch_size}",
            f"{'gradient_accum':{w}}: {self.gradient_accumulation_steps}",
            f"{'max_seq_length':{w}}: {self.max_seq_length}",
            f"{'run_dir':{w}}: {self.run_dir}",
            f"{'log_path':{w}}: {self.log_path}",
        ]
        return "\n".join(lines)
