"""LoRA training pipeline for approximating angular steering.

Goal: train LoRA adapters (Wh + ABh) to reproduce the output of
angular steering (Wh + steering_delta(Wh)), and study the relationship
between the two approaches.

Usage (from angular-steering/ root):
    python -m lora --config lora/configs/example_qwen3b.yaml
    python -m lora --config lora/configs/example_qwen3b.yaml --stage train
    python -m lora --config lora/configs/example_qwen3b.yaml --stage eval

    # Override specific params
    python -m lora --config lora/configs/example_qwen3b.yaml --rank 1
    python -m lora --config lora/configs/example_qwen3b.yaml --rank 8 --epochs 5
"""

from .config import LoRAConfig

__all__ = ["LoRAConfig"]
