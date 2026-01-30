"""Common utilities for PyTorch steering implementation.

This module contains shared data loading, tokenization, and hook utilities
used by both extract_directions.py and generate_responses.py.
"""

import functools
import io
from contextlib import contextmanager
from functools import cache
from typing import Callable, List, Tuple

import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# =============================================================================
# Data Loading Utilities
# =============================================================================


def get_harmful_instructions():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    """Load harmless instructions from Alpaca dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        item["instruction"] for item in dataset["train"] if item["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train[:512], test[:128]


@cache
def get_input_data(
    data_type: str, language_id: str = "en"
) -> Tuple[List[str], List[str]]:
    """Get training and test data."""
    if language_id != "en":
        raise ValueError(f"Only English (en) is supported, got: {language_id}")
    if data_type == "harmful":
        return get_harmful_instructions()
    elif data_type == "harmless":
        return get_harmless_instructions()
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


# =============================================================================
# Tokenization and Hook Utilities
# =============================================================================


def tokenize_instructions_fn(instructions, tokenizer, system_prompt=None):
    """Tokenize instructions using chat template.

    Args:
        instructions: List of instruction strings
        tokenizer: HuggingFace tokenizer with chat template support
        system_prompt: Optional system prompt to prepend to each instruction

    Returns:
        Tokenized inputs as BatchEncoding with input_ids and attention_mask
    """
    inputs = tokenizer.apply_chat_template(
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
        padding=True,
        truncation=False,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


@contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
    module_forward_hooks: List[Tuple[nn.Module, Callable]] = None,
    **kwargs,
):
    """Context manager for temporarily adding forward hooks.

    Args:
        module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
        module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
        **kwargs: Additional keyword arguments passed to hook functions

    Yields:
        None. Hooks are active within the context, removed on exit.
    """
    module_forward_pre_hooks = module_forward_pre_hooks or []
    module_forward_hooks = module_forward_hooks or []
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


# =============================================================================
# Activation Capture Hooks
# =============================================================================


def get_residual_hook(
    cache_key_prefix: str, cache: dict, positions: list, extract_positions: list = None
):
    """Create hook for decoder layers to capture residual stream at resid_post.

    Args:
        cache_key_prefix: Prefix for cache keys (e.g., 'layer_10')
        cache: Dictionary to store captured activations
        positions: List of positions to capture - typically ['post'] for end of layer
        extract_positions: Token positions to extract (default [-1] for last token)

    Returns:
        Hook function that captures layer outputs to cache
    """
    extract_positions = extract_positions or [-1]

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output

        if "post" in positions:
            cache_key = f"{cache_key_prefix}_post"
            if extract_positions == [-1]:
                acts = hidden_states[:, -1:, :].detach().cpu()
            else:
                acts = hidden_states[:, extract_positions, :].detach().cpu()

            if cache_key in cache:
                cache[cache_key] = torch.cat([cache[cache_key], acts], dim=0)
            else:
                cache[cache_key] = acts

    return hook_fn


def get_mlp_input_hook(cache_key: str, cache: dict, extract_positions: list = None):
    """Create pre-hook to capture resid_mid from post_attention_layernorm input.

    Args:
        cache_key: Key to use in cache dictionary (e.g., 'layer_10_mid')
        cache: Dictionary to store captured activations
        extract_positions: Token positions to extract (default [-1] for last token)

    Returns:
        Pre-hook function that captures layer inputs to cache
    """
    extract_positions = extract_positions or [-1]

    def hook_fn(module, input):
        if extract_positions == [-1]:
            acts = input[0][:, -1:, :].detach().cpu()
        else:
            acts = input[0][:, extract_positions, :].detach().cpu()

        if cache_key in cache:
            cache[cache_key] = torch.cat([cache[cache_key], acts], dim=0)
        else:
            cache[cache_key] = acts

    return hook_fn
