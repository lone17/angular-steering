"""LoRA vs Angular Steering analysis module."""

from .activation_analysis import run_activation_analysis
from .weight_analysis import run_weight_analysis


def run_analysis(
    config,
    lora_path,
    directions_file,
    n_samples: int = 20,
    batch_size: int = 4,
    skip_activation: bool = False,
) -> dict:
    """Run weight and (optionally) activation analysis.

    Returns:
        {"weight": [...], "activation": [...] or None}
    """
    weight_rows = run_weight_analysis(
        lora_path=lora_path,
        directions_file=directions_file,
        lora_rank=config.lora_rank,
    )

    activation_rows = None
    if not skip_activation:
        activation_rows = run_activation_analysis(
            config=config,
            lora_path=lora_path,
            directions_file=directions_file,
            n_samples=n_samples,
            batch_size=batch_size,
        )

    return {"weight": weight_rows, "activation": activation_rows}


__all__ = ["run_analysis", "run_weight_analysis", "run_activation_analysis"]
