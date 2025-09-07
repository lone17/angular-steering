from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import argparse
import functools
import random

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .common.utility import (
    add_hooks,
    get_dataset_instructions,
    load_model_and_tokenizer,
    tokenize_instructions_qwen_chat,
)

from .common.enum import GenerationType

UPPERBOUND_MAX_NEW_TOKENS = 16000
DEVICE = "auto"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_NAME = MODEL_ID.split("/")[-1]
RANDOM_SEED = 42
OUTPUT_ROOT = Path("outputs")
OUTPUT_DIR = OUTPUT_ROOT / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTION_POINT = 52


def _get_rotation_args(
    first_directions: Tensor,
    second_directions: Optional[Tensor],
    target_degree: float,
) -> tuple[Tensor | None, Tensor | None]:
    """Compute the rotated component with respect to a 2D subspace and an rotation
    angle."""

    if second_directions is None:
        return None, None

    # first_direction: (batch) x hidden_dim
    # second_directions: (batch) x hidden_dim

    # ensure bases are orthonormal
    b1 = first_directions / first_directions.norm(dim=-1, keepdim=True)
    b2 = (
        second_directions - torch.sum(second_directions * b1, dim=-1, keepdim=True) * b1
    )
    b2 /= b2.norm(dim=-1, keepdim=True)

    theta = np.deg2rad(target_degree)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    proj_matrix = torch.einsum("...i, ...j -> ...ij", b1, b1) + torch.einsum(
        "...i, ...j -> ...ij", b2, b2
    )

    uv = torch.stack([b1.expand_as(b2), b2], dim=-1)  # shape (..., 2)

    # rotate counter-clockwise
    R_theta = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        device=uv.device,
        dtype=uv.dtype,
    )

    rotated_component = (
        uv @ R_theta @ torch.tensor([1, 0], device=uv.device, dtype=uv.dtype)
    )

    return proj_matrix, rotated_component


def get_angular_steering_output_hook(
    first_direction: np.ndarray,
    second_direction: np.ndarray,
    target_degree: float,
    adaptive_mode: int = 1,
) -> Callable[[nn.Module, Tuple[Any, ...], Any], Any]:
    first_dir = torch.from_numpy(first_direction)
    second_dir = torch.from_numpy(second_direction)
    proj_matrix, rotated_component = _get_rotation_args(
        first_directions=first_dir,
        second_directions=second_dir,
        target_degree=target_degree,
    )

    def hook_fn(module, input, output):
        nonlocal first_dir, second_dir, proj_matrix, rotated_component
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output
        first_dir = torch.from_numpy(first_direction)
        second_dir = torch.from_numpy(second_direction)
        proj_matrix = proj_matrix.to(activation)
        rotated_component = rotated_component.to(activation)
        Px = torch.einsum("...i, ...ij -> ...j", activation, proj_matrix)
        scale = Px.norm(dim=-1, keepdim=True)
        if adaptive_mode in {0, 4}:
            activation += -Px + scale * rotated_component
        else:
            if adaptive_mode == 1:
                feature_direction = first_dir
            elif adaptive_mode == 2:
                feature_direction = second_dir
            elif adaptive_mode == 3:
                feature_direction = first_dir
            else:
                raise ValueError(f"Invalid adaptive mode: {adaptive_mode}")
            feature_direction = feature_direction.to(
                device=activation.device, dtype=activation.dtype
            )
            proj_to_feature_direction = activation @ feature_direction
            mask = proj_to_feature_direction > 0
            # activation: batch x seq_len x hidden_dim
            # mask: batch x seq_len
            # scale: batch x seq_len x 1
            # rotated_component: (batch) x seq_len x hidden_dim
            # Px: batch x seq_len x hidden_dim
            activation += mask.unsqueeze(-1) * (scale * rotated_component - Px)
        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def load_candidate_steering_vectors(output_path: Path) -> tuple[Tensor, Tensor]:
    candidate_vectors_path = output_path / "candidate_steering_vectors.pt"
    candidate_vectors_normed_path = output_path / "candidate_steering_vectors_normed.pt"
    if not candidate_vectors_path.exists() or not candidate_vectors_normed_path.exists():
        raise FileNotFoundError()

    candidate_steering_vectors = torch.load(candidate_vectors_path)
    candidate_steering_vectors_normed = torch.load(candidate_vectors_normed_path)
    return candidate_steering_vectors, candidate_steering_vectors_normed


def calculate_basis(
    output_path: Path, extraction_point: int = EXTRACTION_POINT
) -> tuple[np.ndarray, np.ndarray]:
    steering_dirs, _ = load_candidate_steering_vectors(output_path)

    steering_dirs_flatten = steering_dirs.reshape((-1, steering_dirs.shape[-1])).cpu().numpy()
    steering_dir = steering_dirs_flatten[extraction_point]
    steering_dir_np = steering_dir

    first_basis = steering_dir_np / np.linalg.norm(steering_dir_np)
    print(f"First basis: {first_basis}")
    print(f"First basis - shape: {first_basis.shape}")

    pca = PCA()
    pca.fit(steering_dirs_flatten)

    # Get the principal components
    principal_components = pca.components_
    pca_0 = principal_components[0]

    print(f"First principal component: {pca_0}")
    print(f"Shape: {pca_0.shape}")

    # Orthogonalize the pca0 onto the first basis
    projected_pca0_vector = (pca_0 @ first_basis) * first_basis

    second_basis = pca_0 - projected_pca0_vector
    second_basis /= np.linalg.norm(second_basis)

    print(f"Second basis: {second_basis}")
    print(f"Second basis - shape: {second_basis.shape}")

    assert np.dot(first_basis, second_basis) < 1e-2
    return first_basis, second_basis


# -----------------------------
# Generation functions
# -----------------------------


def generate_completions(
    model: AutoModelForCausalLM,
    instructions: List[str],
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    system_prompt: Optional[str] = None,
    fwd_pre_hooks: List[Tuple[nn.Module, Callable]] = [],
    fwd_hooks: List[Tuple[nn.Module, Callable]] = [],
    max_new_tokens: int = UPPERBOUND_MAX_NEW_TOKENS,
) -> List[Dict[str, str]]:
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    completions = []

    tokenized_instructions = tokenize_instructions_fn(
        instructions=instructions, system_prompt=system_prompt
    )
    
    with add_hooks(
        module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
    ), torch.inference_mode():
        generation_toks = model.generate(
            input_ids=tokenized_instructions.input_ids.to(model.device),
            attention_mask=tokenized_instructions.attention_mask.to(model.device),
            **generation_config,
        )

    # Strip the prompt prefix so we only decode new tokens
    gen_tail = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

    for idx, gen in enumerate(gen_tail):
        completions.append(
            {
                "prompt": instructions[idx],  # fixed: use the matching prompt, not [0]
                "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
            }
        )

    return completions


def generate_completions_early_stop(
    model: AutoModelForCausalLM,
    instructions: List[str],
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    system_prompt: Optional[str] = None,
    fwd_pre_hooks: List[Tuple[nn.Module, Callable]] = [],
    fwd_hooks: List[Tuple[nn.Module, Callable]] = [],
    max_new_tokens: int = UPPERBOUND_MAX_NEW_TOKENS,
    disable_token_id: int = 151668,     # stop applying hooks AFTER this id appears
) -> List[Dict[str, str]]:
    """
    Runs HF generate() with your hooks, but disables all hooks as soon as
    the token `disable_token_id` is generated (for any sequence in the batch).
    Generation itself keeps going; only hooks stop being applied.
    """

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # --- shared gating state ---
    shared = {"hooks_enabled": True}

    # Wrappers that make your hooks conditional on the shared flag.
    # For forward-pre hooks: return None when disabled (means "no change").
    def gate_pre(h):
        def wrapped(module, *args, **kwargs):
            if not shared["hooks_enabled"]:
                return None
            return h(module, *args, **kwargs)
        return wrapped

    # For forward hooks: return None when disabled (means "keep original output").
    def gate_post(h):
        def wrapped(module, *args, **kwargs):
            if not shared["hooks_enabled"]:
                return None
            return h(module, *args, **kwargs)
        return wrapped

    gated_pre_hooks  = [(m, gate_pre(h))  for (m, h) in fwd_pre_hooks]
    gated_post_hooks = [(m, gate_post(h)) for (m, h) in fwd_hooks]

    # Custom stopping criteria that *disables hooks* once the token appears.
    class DisableHooksOnToken(StoppingCriteria):
        def __init__(self, token_id: int, shared_state: dict):
            self.token_id = token_id
            self.shared_state = shared_state
            self.tripped = False

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            # input_ids: [batch, cur_len]. Check last generated token.
            if not self.tripped and (input_ids[:, -1] == self.token_id).any():
                self.shared_state["hooks_enabled"] = False
                self.tripped = True
            # Never stop generation; we only toggle hooks.
            return False

    stopping = StoppingCriteriaList([DisableHooksOnToken(disable_token_id, shared)])

    completions = []

    # Your tokenizer fn should return an object with .input_ids and .attention_mask
    tokenized_instructions = tokenize_instructions_fn(
        instructions=instructions, system_prompt=system_prompt
    )
    # Register hooks for the whole generate() call; they self-disable via `shared`.
    with add_hooks(
        module_forward_pre_hooks=gated_pre_hooks,
        module_forward_hooks=gated_post_hooks,
    ), torch.inference_mode():
        generation_toks = model.generate(
            input_ids=tokenized_instructions.input_ids.to(model.device),
            attention_mask=tokenized_instructions.attention_mask.to(model.device),
            stopping_criteria=stopping,   # <-- flips hooks off at the right moment
            **generation_config,
        )

    # Strip the prompt prefix so we only decode new tokens
    gen_tail = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

    for idx, gen in enumerate(gen_tail):
        completions.append(
            {
                "prompt": instructions[idx],  # fixed: use the matching prompt, not [0]
                "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
            }
        )

    return completions


def generate_completions_delay(
    model: AutoModelForCausalLM,
    instructions: List[str],
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    system_prompt: Optional[str] = None,
    fwd_pre_hooks: List[Tuple[nn.Module, Callable]] = [],
    fwd_hooks: List[Tuple[nn.Module, Callable]] = [],
    max_new_tokens: int = UPPERBOUND_MAX_NEW_TOKENS,
    start_token_id: int = 151667,      # turn ON window after this appears
    end_token_id: int = 151668,        # turn OFF after this appears
    start_delay_n: int = 100,            # number of tokens to wait after start before enabling hooks
) -> List[Dict[str, str]]:
    """
    Hooks are applied only in the window:
        (start_token_id) --[wait start_delay_n tokens]-->  (hooks ON)  ... until (end_token_id) --> (hooks OFF)
    Hooks are OFF during prefill and until the delayed window begins. Generation itself never stops due to this gating.
    """

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }


    # ---- shared gate state (global across the batch) ----
    gate = {
        "enabled": False,          # whether hooks should currently run
        "seen_start": False,       # have we seen start_token_id yet?
        "seen_end": False,         # have we seen end_token_id yet?
        "delay_count": 0,          # how many tokens since start we have generated
    }

    # Wrap your hooks to obey the gate
    def gate_pre(h):
        def wrapped(module, *args, **kwargs):
            if not gate["enabled"]:
                return None  # no-op for pre-hooks
            return h(module, *args, **kwargs)
        return wrapped

    def gate_post(h):
        def wrapped(module, *args, **kwargs):
            if not gate["enabled"]:
                return None  # no-op for post-hooks (keeps original output)
            return h(module, *args, **kwargs)
        return wrapped

    gated_pre_hooks  = [(m, gate_pre(h))  for (m, h) in fwd_pre_hooks]
    gated_post_hooks = [(m, gate_post(h)) for (m, h) in fwd_hooks]

    # StoppingCriteria that *doesn't* stop generation; it just updates the gate each step.
    class ToggleHooksWithDelay(StoppingCriteria):
        def __init__(self, start_id: int, end_id: int, state: dict, delay_n: int):
            self.start = start_id
            self.end   = end_id
            self.state = state
            self.delay = max(int(delay_n), 0)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            # input_ids: [batch, cur_len]; check last generated token in this step
            last_ids = input_ids[:, -1]

            # If we see the start token for the first time, arm the delay window
            if not self.state["seen_start"] and (last_ids == self.start).any():
                self.state["seen_start"] = True
                self.state["delay_count"] = 0
                self.state["enabled"] = False  # still OFF until delay elapses

            # If we already saw start and haven't seen end, step the delay counter
            if self.state["seen_start"] and not self.state["seen_end"]:
                # Only increment while not yet enabled
                if not self.state["enabled"]:
                    self.state["delay_count"] += 1
                    if self.state["delay_count"] >= self.delay:
                        self.state["enabled"] = True  # turn hooks ON after waiting delay_n tokens

            # If we see the end token, turn hooks OFF
            if not self.state["seen_end"] and (last_ids == self.end).any():
                self.state["seen_end"] = True
                self.state["enabled"] = False

            return False  # never stop generation here

    stopping = StoppingCriteriaList([ToggleHooksWithDelay(start_token_id, end_token_id, gate, start_delay_n)])

    completions = []

    tokenized = tokenize_instructions_fn(instructions=instructions)

    # Register hooks for the whole generate() call; wrappers consult `gate`
    with add_hooks(
        module_forward_pre_hooks=gated_pre_hooks,
        module_forward_hooks=gated_post_hooks,
    ):
        with torch.inference_mode():
            generation_toks = model.generate(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
                stopping_criteria=stopping,  # <-- flips the gate on/off with delay
                **generation_config,
            )

    # Keep only the generated tail
    gen_tail = generation_toks[:, tokenized.input_ids.shape[-1]:]

    for i, gen in enumerate(gen_tail):
        completions.append({
            "prompt": instructions[i],
            "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
        })
    return completions

# -----------------------------
# Utilities
# -----------------------------


def set_reproducibility(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get activations from language model")
    parser.add_argument(
        "--model_id", type=str, default=MODEL_ID, help="Model name or path"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Device to run the model on"
    )
    parser.add_argument(
        "--upperbound_max_new_tokens",
        type=int,
        default=UPPERBOUND_MAX_NEW_TOKENS,
        help="Max number of tokens to generate (-1 for default)",
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
        "--extraction_point",
        type=int,
        default=52,
        help="Layers to extract activations from (default: 52 for Qwen3-4B)",
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
        default=5,
        help="End index for processing instructions",
    )
    parser.add_argument(
        "--target_angle",
        type=int,
        nargs="+",
        default=[90, 120, 180, 200, 240, 270],
        help="Target angles for steering",
    )
    parser.add_argument(
        "--generation_type",
        type=GenerationType,
        default=GenerationType.DEFAULT,
        help="Type of generation to use (default, early_stop or delay)",
    )
    return parser.parse_args()


def main() -> None:
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
    set_reproducibility(seed=args.random_seed)

    _, instructions_test = get_dataset_instructions(
        random_seed=args.random_seed
    )

    first_basis, second_basis = calculate_basis(args.output_dir, extraction_point=args.extraction_point)

    module_dict = dict(model.named_modules())
    module_names = args.module_names
    module_name_list = [f"model.layers.{i}.{name}" for i in range(model.config.num_hidden_layers) for name in module_names]

    print("STARTING ANGULAR STEERING EXPERIMENTS")
    print("=" * 100)

    answer_dict: Dict[int, Dict[str, Dict[str, str]]] = {}

    if args.generation_type == GenerationType.DEFAULT:
        generate_completions_fn = generate_completions
    elif args.generation_type == GenerationType.EARLY_STOP:
        generate_completions_fn = generate_completions_early_stop
    elif args.generation_type == GenerationType.DELAY:
        generate_completions_fn = generate_completions_delay
    else:   
        raise ValueError(f"Invalid generation type: {args.generation_type}")
    
    for target_degree in args.target_angle:
        answer_dict[target_degree] = {} # type: ignore
        output_hooks = [
            (
                module_dict[module_name],
                get_angular_steering_output_hook(
                    first_direction=first_basis,
                    second_direction=second_basis,
                    target_degree=target_degree,
                    adaptive_mode=0,
                ),
            )
            for module_name in module_name_list
        ]

        for i in range(args.start_index, args.end_index):
            print(f"\nSteering at {target_degree} degrees")
            print("-" * 100)
            print("QUESTION:\n", instructions_test[i])
            print("ANSWER:")
            completions = generate_completions_fn(
                model=model,
                instructions=[instructions_test[i]],
                tokenizer=tokenizer,
                tokenize_instructions_fn=tokenize_instructions_fn,
                system_prompt=None,
                # system_prompt="Answer in the most angry, frustrated and aggressive way possible.",
                # fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=output_hooks,
                max_new_tokens=args.upperbound_max_new_tokens,
            )
            print(completions[0]["response"][:500])
            print(f"FIRST END THINK TOKEN POSITION: {completions[0]['response'].find('</think>')}")
            print(f"LAST END THINK TOKEN POSITION: {completions[0]['response'].rfind('</think>')}")
            print("_" * 100)
        answer_dict[target_degree][instructions_test[i]] = completions


if __name__ == "__main__":
    main()