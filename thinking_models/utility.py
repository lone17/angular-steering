import contextlib
import random
import requests
import torch
import functools

from jaxtyping import Int
from pathlib import Path
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Callable, Generator

# Constants
DEVICE = "cuda"

def load_model_and_tokenizer(
    model_id: str, device: str = DEVICE, dtype: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: list[str],
    system_prompt: Optional[str] = None,
    enable_thinking: bool = True,
) -> Int[Tensor, "batch_size seq_len"]:
    """
    Tokenize instructions for Qwen chat model.

    """
    prompts = tokenizer.apply_chat_template(
        [
            (
                [{"role": "user", "content": instruction}]
                if system_prompt is None
                else [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]
            )
            for instruction in instructions
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def get_dataset_instructions(random_seed=42) -> tuple[list[str], list[str]]:
    if not Path("messages.py").exists():
        url = "https://raw.githubusercontent.com/cvenhoff/steering-thinking-llms/refs/heads/main/messages/messages.py"

        response = requests.get(url)
        # Save to file
        with open("messages.py", "w") as f:
            f.write(response.text)

    # Load the messages
    assert Path("messages.py").exists()
    from messages import eval_messages, messages

    train_contents = [msg["content"] for msg in messages]
    eval_contents = [msg["content"] for msg in eval_messages]

    # Shuffle the messages
    random.seed(random_seed)
    random.shuffle(train_contents)
    random.shuffle(eval_contents)

    return train_contents, eval_contents


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