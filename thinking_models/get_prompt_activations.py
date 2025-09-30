import argparse
import functools
from pathlib import Path
from typing import Any, Callable

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utility import (
    add_hooks,
    get_dataset_instructions,
    load_model_and_tokenizer,
    tokenize_instructions_qwen_chat,
)

# Constants
TRAIN_BATCH_SIZE = 512
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_NAME = MODEL_ID.split("/")[-1]
RANDOM_SEED = 42
OUTPUT_ROOT = Path("outputs")
OUTPUT_DIR = OUTPUT_ROOT / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_target_module_paths(module_names: list[str], layers: list[int]) -> list[str]:
    return [
        f"model.layers.{layer}.{module_name}"
        for layer in layers
        for module_name in module_names
    ]


def get_activations_pre_hook(
    module_name: str,
    cache: dict[str, Float[Tensor, "batch token d_model"]],
    positions: list[int],
    offload_to_cpu: bool = False,
) -> Callable:
    def hook_fn(module, input):
        activation: Float[Tensor, "batch token d_model"] = input[0]

        # Extract only the required positions to minimize memory usage
        if positions == [-1]:
            # For generation, only capture the last token position
            selected_activation = activation[:, -1:, :].clone()
        else:
            selected_activation = activation[:, positions, :].clone()

        if offload_to_cpu:
            selected_activation = selected_activation.cpu()

        if module_name not in cache:
            cache[module_name] = selected_activation
        else:
            cache[module_name] = torch.cat(
                (cache[module_name], selected_activation), dim=1
            )

    return hook_fn


def get_prompt_token_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    prompt: str,
    module_names: list[str] = ["input_layernorm", "post_attention_layernorm"],
    layers: list[int] | None = None,
    offload_to_cpu: bool = False,
) -> dict[str, Float[Tensor, "batch token d_model"]]:
    if layers is None:
        # Get number of layers from model config
        if hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, "n_layers"):
            num_layers = model.config.n_layers
        else:
            raise ValueError("Could not determine number of layers from model config")
        layers = list(range(num_layers))

    tokens = tokenize_instructions_fn(instructions=[prompt], enable_thinking=True)
    print(f"Number of input tokens: {tokens.input_ids.shape[1]}")

    module_dict = dict(model.named_modules())
    target_module_paths = get_target_module_paths(module_names, layers)
    step_activations: dict[str, Float[Tensor, "batch token d_model"]] = {} # dict[module_path, activation]
    fwd_pre_hooks = [
        (
            module_dict[module_path],
            get_activations_pre_hook(
                module_path,
                step_activations,
                positions=range(tokens.input_ids.shape[-1]),
                offload_to_cpu=offload_to_cpu,
            ),
        )
        for module_path in target_module_paths
    ]

    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks,
        module_forward_hooks=[],
    ):
        with torch.no_grad():
            model.forward(
                tokens.input_ids.to(model.device),
                tokens.attention_mask.to(model.device),
            )

    # Number of prompt tokens must match the second dim of per-module tensor
    assert tokens.input_ids.shape[-1] == step_activations[target_module_paths[0]].shape[1]
    return step_activations


def preprocess_instructions(instructions: list[str]) -> list[str]:
    return [ins + " Think fast and briefly." for ins in instructions]


# Save mean activations to disk for backup
def save_tensor(
    tensor: Tensor | dict[str, Any],
    file_path: Path,
) -> None:
    torch.save(tensor, file_path)
    print(f"Saved {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Get activations from language model")
    parser.add_argument(
        "--model_id", type=str, default=MODEL_ID, help="Model name or path"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Device to run the model on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help="Number of instructions to process",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--module_names",
        type=str,
        nargs="+",
        default=["input_layernorm", "post_attention_layernorm"],
        help="Module names to extract activations from",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to extract activations from (default: all layers)",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Start index for processing instructions",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        type=int,
        default=None,
        help="End index for processing instructions",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offload activations to CPU",
    )
    parser.add_argument(
        "--think_fast",
        action="store_true",
        help="Think fast and briefly",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer with args
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)

    tokenize_instructions_fn = functools.partial(
        tokenize_instructions_qwen_chat,
        tokenizer=tokenizer,
        enable_thinking=True,
    )

    # Prepare dataset
    instructions_train, instructions_test = get_dataset_instructions(
        random_seed=args.random_seed
    )

    if args.think_fast:
        instructions_train = preprocess_instructions(instructions_train)
        instructions_test = preprocess_instructions(instructions_test)

    # Get activations and save output to disk
    if args.end_index is None:
        args.end_index = args.batch_size

    args.end_index = min(args.end_index, len(instructions_train))

    sequences_activations = []
    output_dir = args.output_dir / "prompt"
    output_dir.mkdir(exist_ok=True)

    subset_instructions = instructions_train[: args.batch_size]
    for i in range(args.start_index, args.end_index):
        instruction = subset_instructions[i]
        if not args.think_fast:
            file_path = output_dir / f"prompt_activations_{i}.pt"
        else:
            file_path = output_dir / f"think_fast_prompt_activations_{i}.pt"
        # if file_path.exists():
        #     print(f"Skipping {file_path} because it already exists")
        #     continue

        print(f"Processing instruction {i}/{args.batch_size}:")
        seq_activations = get_prompt_token_activations(
            model,
            tokenizer,
            tokenize_instructions_fn,
            prompt=instruction,
            module_names=args.module_names,
            offload_to_cpu=args.offload_to_cpu,
        )
        sequences_activations.append(seq_activations)

        save_tensor(seq_activations, file_path)
        print("--------------------------------")
