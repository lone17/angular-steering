from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import argparse
import functools
import math
import random
import json

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

from common.utility import (
    add_hooks,
    get_dataset_instructions,
    load_model_and_tokenizer,
    tokenize_instructions_qwen_chat,
)

from common.enum import GenerationType
from common.schema import (
    DisableHooksOnToken,
    GenerationConfigDict,
    ToggleHooksWithDelay,
)


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


def calculate_basis(
    pre_computed_steering_vectors_path: Path,
    extraction_point: int = EXTRACTION_POINT
) -> tuple[np.ndarray, np.ndarray]:

    if not pre_computed_steering_vectors_path.exists():
        raise FileNotFoundError(pre_computed_steering_vectors_path)

    steering_vectors = torch.load(pre_computed_steering_vectors_path)

    steering_vectors_flatten = steering_vectors.reshape(
        (-1, steering_vectors.shape[-1])
    ).cpu().numpy()
    steering_vector = steering_vectors_flatten[extraction_point]
    steering_vector_np = steering_vector

    first_basis = steering_vector_np / np.linalg.norm(steering_vector_np)
    print(f"First basis: {first_basis}")
    print(f"First basis - shape: {first_basis.shape}")

    pca = PCA()
    pca.fit(steering_vectors_flatten)

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
    generation_config = GenerationConfigDict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

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
    print(f"Number of generated tokens: {len(gen_tail[0])}")

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

    generation_config = GenerationConfigDict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

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
    print(f"Number of generated tokens: {len(gen_tail[0])}")

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

    generation_config = GenerationConfigDict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )


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
    print(f"Number of generated tokens: {len(gen_tail[0])}")

    for i, gen in enumerate(gen_tail):
        completions.append({
            "prompt": instructions[i],
            "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
        })
    return completions


def _wrap_to_pm180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _nearest_angle(current_angle, target_angles) -> float:
    return min(target_angles, key=lambda a: abs(_wrap_to_pm180(current_angle - a)))


def make_dynamic_angular_hook_using_rotation_args(
    first_direction,
    second_direction,
    gate: dict,
    target_angles: List[float] = [90.0, 270.0],
    adaptive_mode: int = 0,
):
    first_dir_base = torch.from_numpy(first_direction)
    second_dir_base = torch.from_numpy(second_direction)

    with torch.no_grad():
        _, b1_cpu = _get_rotation_args(first_dir_base, second_dir_base, target_degree=0.0)
        _, b2_cpu = _get_rotation_args(first_dir_base, second_dir_base, target_degree=90.0)

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            act = output[0]
            rest = output[1:]
        else:
            act = output
            rest = None

        if not gate.get("enabled", False):
            return output

        b1 = b1_cpu.to(act)
        b2 = b2_cpu.to(act)

        alpha = (act * b1).sum(dim=-1)
        beta  = (act * b2).sum(dim=-1)

        if gate.get("mode", "measure") == "measure":
            alpha_last = alpha[:, -1].mean().item()
            beta_last  = beta[:, -1].mean().item()
            theta_now  = math.degrees(math.atan2(beta_last, alpha_last))  # [-180, 180)

            gate["angle_cur_measured"] = theta_now
            # Choose target angle
            if gate.get("target_mode", "nearest_angle") == "nearest_angle":
                gate["angle_target"] = _nearest_angle(theta_now, target_angles)
            else:
                gate["angle_target"] = float(gate.get("fixed_target_deg", 90.0))
            gate["angle_cur"] = theta_now
            gate["mode"] = "apply"  # start applying from next step
            return output

        # APPLY MODE: build projector and rotated vector for the CURRENT angle
        angle_deg = float(gate["angle_cur"])
        proj_matrix, rotated_component = _get_rotation_args(
            first_directions=b1,
            second_directions=b2,
            target_degree=angle_deg,
        )
        proj_matrix      = proj_matrix.to(act)         # [..., D, D]
        rotated_component= rotated_component.to(act)   # [..., D]

        # Project to plane via P x and preserve in-plane norm
        Px    = torch.einsum("...i, ...ij -> ...j", act, proj_matrix)  # [B,S,D]
        scale = Px.norm(dim=-1, keepdim=True)                          # [B,S,1]

        if adaptive_mode in {0, 4}:
            act = act + (-Px + scale * rotated_component)
        else:
            # Choose feature direction for gating
            if adaptive_mode == 1:
                feature_direction = b1
            elif adaptive_mode == 2:
                feature_direction = b2
            elif adaptive_mode == 3:
                feature_direction = b1
            else:
                raise ValueError(f"Invalid adaptive_mode: {adaptive_mode}")
            proj_to_feature = (act * feature_direction).sum(dim=-1)  # [B,S]
            mask = (proj_to_feature > 0).unsqueeze(-1)               # [B,S,1]
            act = act + mask * (scale * rotated_component - Px)

        if rest is None:
            return act
        else:
            return (act, *rest)

    return hook_fn


class ToggleHooksWithAngleSchedule(StoppingCriteria):
    """
    Does not stop generation.
    - Turns hooks ON after `start_id`, with a delay of `delay_n` tokens.
    - On the first enabled step, the hook is in "measure" mode (no rotation).
    - Then, every `step_every` steps, moves `angle_cur` by ±`step_deg` toward `angle_target`.
    - Turns hooks OFF after `end_id`.
    """
    def __init__(
        self,
        start_id: int,
        end_id: int,
        state: dict,
        delay_n: int,
        step_deg: float = 1.0,
        step_every: int = 1,
    ):
        self.start = start_id
        self.end   = end_id
        self.st    = state
        self.delay = max(int(delay_n), 0)
        self.step_deg = float(step_deg)
        self.step_every = max(int(step_every), 1)
        self._tick = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -1]

        # Observe start token
        if not self.st.get("seen_start", False) and (last_ids == self.start).any():
            self.st["seen_start"]  = True
            self.st["delay_count"] = 0
            self.st["enabled"]     = False
            self.st["mode"]        = "measure"  # first ON step will only measure

        # Handle delay and scheduling while before end
        if self.st.get("seen_start", False) and not self.st.get("seen_end", False):
            if not self.st.get("enabled", False):
                self.st["delay_count"] += 1
                if self.st["delay_count"] >= self.delay:
                    self.st["enabled"] = True
                    self._tick = 0

            # If enabled and already in "apply" mode, move angle toward target
            if self.st.get("enabled", False) and self.st.get("mode") == "apply" and "angle_cur" in self.st:
                self._tick += 1
                if self._tick % self.step_every == 0:
                    err = _wrap_to_pm180(self.st["angle_target"] - self.st["angle_cur"])
                    step = max(-self.step_deg, min(self.step_deg, err))
                    self.st["angle_cur"] += step
                    # print("Current angle:", self.st["angle_cur"])

        # Observe end token
        if not self.st.get("seen_end", False) and (last_ids == self.end).any():
            self.st["seen_end"] = True
            self.st["enabled"]  = False
            self.st["mode"]     = "idle"

        return False  # never stop generation here


def generate_completions_scheduler(
    model,
    instructions: List[str],
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable,
    system_prompt: Optional[str],
    module_dict: Dict[str, torch.nn.Module],
    module_names: Sequence[str],                # list of module names to hook
    first_direction,                            # np.ndarray[D] or torch.Tensor[D]
    second_direction,                           # np.ndarray[D] or torch.Tensor[D]
    *,
    max_new_tokens: int = UPPERBOUND_MAX_NEW_TOKENS,
    start_token_id: int = 151667,               # <think>
    end_token_id: int   = 151668,               # </think>
    start_delay_n: int  = 100,                  # wait N tokens after <think>
    step_deg: float     = 1.0,                  # move ±1 degree per update
    step_every: int     = 1,                    # update every k tokens
    target_mode: str    = "nearest_angle",      # or "fixed"
    target_angles: List[float] = [90.0, 270.0], # used if target_mode == "nearest_angle"
    fixed_target_deg: float = 90.0,             # used if target_mode == "fixed"
    adaptive_mode: int = 0,                     # pass-through of your gating semantics
):
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    # Shared gate/state
    gate = {
        "enabled": False,
        "seen_start": False,
        "seen_end": False,
        "delay_count": 0,
        "mode": "idle",                 # "idle" -> "measure" -> "apply"
        "angle_cur": None,
        "angle_target": None,
        "target_mode": target_mode,
        "fixed_target_deg": fixed_target_deg,
    }

    # Build one hook per module name
    fwd_hooks = []
    for name in module_names:
        if name not in module_dict:
            continue
        hook = make_dynamic_angular_hook_using_rotation_args(
            first_direction=first_direction,
            second_direction=second_direction,
            gate=gate,
            target_angles=target_angles,
            adaptive_mode=adaptive_mode,
        )
        fwd_hooks.append((module_dict[name], hook))

    # Scheduler / gating controller
    stopping = StoppingCriteriaList([
        ToggleHooksWithAngleSchedule(
            start_id=start_token_id,
            end_id=end_token_id,
            state=gate,
            delay_n=start_delay_n,
            step_deg=step_deg,
            step_every=step_every,
        )
    ])

    # Tokenize
    tokenized = tokenize_instructions_fn(instructions=instructions, system_prompt=system_prompt)

    # Generate with hooks
    completions = []
    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
        with torch.inference_mode():
            out = model.generate(
                input_ids=tokenized.input_ids.to(model.device),
                attention_mask=tokenized.attention_mask.to(model.device),
                stopping_criteria=stopping,
                **generation_config,
            )
    print("Target degree:", gate["angle_target"])
    # Decode only new tokens
    gen_tail = out[:, tokenized.input_ids.shape[-1]:]
    for i, gen in enumerate(gen_tail):
        completions.append({
            "prompt": instructions[i],
            "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
            "target_degree": gate["angle_target"],
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
        "--steering_vectors_path",
        type=Path,
        help="Path to the file of pre-computed candidate steering vectors",
    )
    parser.add_argument(
        "--chosen_extraction_point",
        type=int,
        default=EXTRACTION_POINT,
        help="The index of the extraction point corresponding to the chosen steering vector",
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
        default=[90, 180, 200, 240, 270],
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

    first_basis, second_basis = calculate_basis(
        pre_computed_steering_vectors_path=Path(args.steering_vectors_path),
        extraction_point=args.chosen_extraction_point
    )

    module_dict = dict(model.named_modules())
    module_names = args.module_names
    module_name_list = []
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        for name in module_names:
            if name != "input_layernorm":
                module_name_list.append(f"model.layers.{i}.{name}")
            elif i < num_layers - 1:
                module_name_list.append(f"model.layers.{i+1}.{name}")
            else:
                continue

        # module_name_list = [
        #     f"model.layers.{i}.{name}" for i in range(model.config.num_hidden_layers) for name in module_names
        # ]

    print("STARTING ANGULAR STEERING EXPERIMENTS")
    print("=" * 100)

    answer_dict: Dict[int, Dict[str, List[Dict[str, str]]]] = {}

    match args.generation_type:
        case GenerationType.DEFAULT:
            generate_completions_fn = generate_completions
        case GenerationType.EARLY_STOP:
            generate_completions_fn = generate_completions_early_stop
        case GenerationType.DELAY:
            generate_completions_fn = generate_completions_delay
        case GenerationType.SCHEDULER:
            pass


    mapping = {
        GenerationType.DEFAULT: generate_completions,
        GenerationType.EARLY_STOP: generate_completions_early_stop,
        GenerationType.DELAY: generate_completions_delay,
    }

    match args.generation_type:
        case GenerationType.SCHEDULER:
            for i in range(args.start_index, args.end_index):
                completions = generate_completions_scheduler(
                model=model,
                instructions=[instructions_test[i]],
                tokenizer=tokenizer,
                tokenize_instructions_fn=tokenize_instructions_fn,
                system_prompt=None,
                max_new_tokens=args.upperbound_max_new_tokens,
                module_dict=module_dict,
                module_names=module_name_list,
                first_direction=first_basis,
                second_direction=second_basis,
                start_token_id=151667,    # <think>
                end_token_id=151668,      # </think>
                start_delay_n=100,
                step_deg=1.0,
                step_every=1,
                target_mode="nearest_angle",
                target_angles=args.target_angle,
                adaptive_mode=1,
            )
                print(f"\nSteering at {completions[0]['target_degree']} degrees")
                print("-" * 100)
                print("QUESTION:\n", instructions_test[i])
                print("ANSWER:")
                print(completions[0]["response"][:1500])
                print(f"FIRST END THINK TOKEN POSITION: {completions[0]['response'].find('</think>')}")
                print(f"LAST END THINK TOKEN POSITION: {completions[0]['response'].rfind('</think>')}")
                print("_" * 100)
                target_degree = completions[0]['target_degree']
                if target_degree not in answer_dict:
                    answer_dict[target_degree] = {}
                answer_dict[target_degree][i] = completions

        case GenerationType.DEFAULT | GenerationType.EARLY_STOP | GenerationType.DELAY:
            generate_completions_fn = mapping[args.generation_type]
            for target_degree in args.target_angle:
                answer_dict[target_degree] = {} # type: ignore
                output_hooks = [
                    (
                        module_dict[module_name],
                        get_angular_steering_output_hook(
                            first_direction=first_basis,
                            second_direction=second_basis,
                            target_degree=target_degree,
                            adaptive_mode=1,
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
                        # fwd_pre_hooks=fwd_pre_hooks,
                        fwd_hooks=output_hooks,
                        max_new_tokens=args.upperbound_max_new_tokens,
                    )
                    print(completions[0]["response"][:1500])
                    print(f"FIRST END THINK TOKEN POSITION: {completions[0]['response'].find('</think>')}")
                    print(f"LAST END THINK TOKEN POSITION: {completions[0]['response'].rfind('</think>')}")
                    print("_" * 100)
                    answer_dict[target_degree][i] = completions
        case _:
            raise ValueError(f"Invalid generation type: {args.generation_type}")

    # Save answer_dict to json
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"angular_steering_{args.generation_type}.json", "w") as f:
        json.dump(answer_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
