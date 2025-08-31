import contextlib
import functools
import time
import random
import requests
from pathlib import Path
from typing import Callable, Generator
from dataclasses import dataclass, asdict

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


random.seed(42)

# Constants
TRAIN_BATCH_SIZE = 32
UPPERBOUND_MAX_NEW_TOKENS = 3000
DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen3-4B"
TOKENIZER_NAME = "Qwen/Qwen3-4B"


@dataclass
class SequenceActivations:
    """
    A dataclass to store the sequence's activations of all tokens along all layers.
    """
    prompt: str
    response: str
    generated_token_ids: list[int]
    token_activations: dict[str, Float[Tensor, "batch token d_model"]] # dict[module_path, activation]


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: list[tuple[torch.nn.Module, Callable]],
    module_forward_hooks: list[tuple[torch.nn.Module, Callable]],
    **kwargs,
) -> Generator[None, None, None]:
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


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
    offload_to_cpu: bool = True,
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

def get_generated_tokens_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    prompt: str,
    max_new_tokens: int = -1,
    module_names: list[str] = ["input_layernorm", "post_attention_layernorm"],
    layers: list[int] | None = None,
) -> SequenceActivations:
    if layers is None:
        # Get number of layers from model config
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layers'):
            num_layers = model.config.n_layers
        else:
            raise ValueError("Could not determine number of layers from model config")
        layers = list(range(num_layers))

    if max_new_tokens == -1:
        max_new_tokens = UPPERBOUND_MAX_NEW_TOKENS
    print(f"Max new tokens: {max_new_tokens}")

    tokens = tokenize_instructions_fn(instructions=[prompt], enable_thinking=True)
    print(f"Number of input tokens: {tokens.input_ids.shape[1]}")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    
    module_dict = dict(model.named_modules())
    target_module_paths = get_target_module_paths(module_names, layers)
    step_activations: dict[str, Float[Tensor, "batch token d_model"]] = {}
    fwd_pre_hooks = [
        (
            module_dict[module_path],
            get_activations_pre_hook(module_path, step_activations, positions=[-1])
        )
        for module_path in target_module_paths
    ]

    start_gen_token_pos = tokens.input_ids.shape[-1]

    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks,
        module_forward_hooks=[],
    ):
        with torch.no_grad():  # Disable gradient computation to save memory
            generated_tokens = model.generate(
                input_ids=tokens.input_ids.to(model.device),
                attention_mask=tokens.attention_mask.to(model.device),
                generation_config=generation_config,
            )

        generated_tokens = generated_tokens[
            :, start_gen_token_pos :
        ]

        # Activations are already collected properly during generation
        # No need to post-process them since we're only capturing new tokens

        return SequenceActivations(
            prompt=prompt,
            response=tokenizer.decode(generated_tokens[0]),
            generated_token_ids=generated_tokens,
            token_activations=step_activations,
        )


def load_model_and_tokenizer(model_name: str, tokenizer_name: str, device: str = DEVICE) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return model, tokenizer


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: list[str],
    enable_thinking: bool = True,
) -> Int[Tensor, "batch_size seq_len"]:
    """
    Tokenize instructions for Qwen chat model.

    """
    prompts = []
    for instruction in instructions:
        message = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)
    
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def get_dataset_instructions() -> tuple[list[str], list[str]]:
    url = "https://raw.githubusercontent.com/cvenhoff/steering-thinking-llms/refs/heads/main/messages/messages.py"

    response = requests.get(url)
    # Save to file
    with open("messages.py", "w") as f:
        f.write(response.text)
    
    # Load the messages
    assert Path("messages.py").exists()
    from messages import messages, eval_messages

    train_contents = [msg["content"] for msg in messages]
    eval_contents = [msg["content"] for msg in eval_messages]

    # Shuffle the messages
    random.shuffle(train_contents)
    random.shuffle(eval_contents)

    return train_contents, eval_contents

def preprocess_instructions(instructions: list[str]) -> list[str]:
    return [ins + " Think fast and briefly." for ins in instructions]


# Save mean activations to disk for backup
def save_tensor(
    tensor: Tensor,
    file_path: Path,
) -> None:
    torch.save(tensor, file_path)
    print(f"Saved {file_path}")


if __name__=="__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, TOKENIZER_NAME)

    tokenize_instructions_fn = functools.partial(
        tokenize_instructions_qwen_chat,
        tokenizer=tokenizer,
        enable_thinking=True,
    )

    # Prepare dataset
    instructions_train, instructions_test = get_dataset_instructions()
    instructions_train = preprocess_instructions(instructions_train)
    instructions_test = preprocess_instructions(instructions_test)

    # Get activations and save output to disk
    sequences_activations = []
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    subset_instructions = instructions_train[:TRAIN_BATCH_SIZE]
    for i, instruction in enumerate(subset_instructions):
        file_path = output_dir / f"sequences_activations_{i}.pt"
        if file_path.exists():
            print(f"Skipping {file_path} because it already exists")
            continue

        print(f"Processing instruction {i}/{TRAIN_BATCH_SIZE}:")
        seq_activations = get_generated_tokens_activations(
            model,
            tokenizer,
            tokenize_instructions_fn,
            prompt=instruction,
            max_new_tokens=-1,
            module_names=["input_layernorm", "post_attention_layernorm"],
        )
        sequences_activations.append(seq_activations)

        save_tensor(asdict(seq_activations), file_path)
        print("--------------------------------")