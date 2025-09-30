from __future__ import annotations

import argparse
import functools
import json
import logging
import random
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import cache

import numpy as np
import torch
import torch.nn as nn
from common.enum import GenerationType
from common.schema import (
    DisableHooksOnToken,
    GenerationConfigDict,
    ToggleHooksWithDelay,
)
from common.utility import (
    add_hooks,
    get_dataset_instructions,
    load_model_and_tokenizer,
    tokenize_instructions_qwen_chat,
)
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

UPPERBOUND_MAX_NEW_TOKENS = 16000
DEVICE = "auto"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_NAME = MODEL_ID.split("/")[-1]
RANDOM_SEED = 42
OUTPUT_ROOT = Path("outputs")
EXTRACTION_POINT = 52


ROTATION_ARGS_CACHE: Dict[Tuple[Tuple, Tuple, float], Tuple[Tensor, Tensor]] = {}


def _get_rotation_args(
    first_directions: Tensor,
    second_directions: Optional[Tensor],
    target_degree: float,
) -> tuple[Tensor | None, Tensor | None]:
    """Compute the rotated component with respect to a 2D subspace and an rotation
    angle."""
    # first_direction: (batch) x hidden_dim
    # second_directions: (batch) x hidden_dim

    if second_directions is None:
        return None, None

    hash_first = tuple(first_directions.cpu().numpy().tolist())
    hash_second = tuple(second_directions.cpu().numpy().tolist())
    cache_key = (hash_first, hash_second, target_degree)
    if cache_key in ROTATION_ARGS_CACHE:
        return ROTATION_ARGS_CACHE[cache_key]

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

    ROTATION_ARGS_CACHE[cache_key] = (proj_matrix, rotated_component)

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
        first_dir = first_dir.to(activation)
        second_dir = second_dir.to(activation)
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
    pre_computed_steering_vectors_path: Path, extraction_point: int = EXTRACTION_POINT
) -> tuple[np.ndarray, np.ndarray]:

    if not pre_computed_steering_vectors_path.exists():
        raise FileNotFoundError(pre_computed_steering_vectors_path)

    steering_vectors = torch.load(pre_computed_steering_vectors_path)

    steering_vectors_flatten = (
        steering_vectors.reshape((-1, steering_vectors.shape[-1])).cpu().numpy()
    )
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

    with (
        add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
        ),
        torch.inference_mode(),
    ):
        generation_toks = model.generate(
            input_ids=tokenized_instructions.input_ids.to(model.device),
            attention_mask=tokenized_instructions.attention_mask.to(model.device),
            **generation_config,
        )

    # Strip the prompt prefix so we only decode new tokens
    gen_tail = generation_toks[:, tokenized_instructions.input_ids.shape[-1] :]
    logger.info(f"Number of generated tokens: {len(gen_tail[0])}")

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
    disable_token_id: int = 151668,  # stop applying hooks AFTER this id appears
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

    gated_pre_hooks = [(m, gate_pre(h)) for (m, h) in fwd_pre_hooks]
    gated_post_hooks = [(m, gate_post(h)) for (m, h) in fwd_hooks]

    stopping = StoppingCriteriaList([DisableHooksOnToken(disable_token_id, shared)])

    completions = []

    # Your tokenizer fn should return an object with .input_ids and .attention_mask
    tokenized_instructions = tokenize_instructions_fn(
        instructions=instructions, system_prompt=system_prompt
    )
    # Register hooks for the whole generate() call; they self-disable via `shared`.
    with (
        add_hooks(
            module_forward_pre_hooks=gated_pre_hooks,
            module_forward_hooks=gated_post_hooks,
        ),
        torch.inference_mode(),
    ):
        generation_toks = model.generate(
            input_ids=tokenized_instructions.input_ids.to(model.device),
            attention_mask=tokenized_instructions.attention_mask.to(model.device),
            stopping_criteria=stopping,  # <-- flips hooks off at the right moment
            **generation_config,
        )

    # Strip the prompt prefix so we only decode new tokens
    gen_tail = generation_toks[:, tokenized_instructions.input_ids.shape[-1] :]
    logger.info(f"Number of generated tokens: {len(gen_tail[0])}")

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
    start_token_id: int = 151667,  # turn ON window after this appears
    end_token_id: int = 151668,  # turn OFF after this appears
    start_delay_n: int = 100,  # number of tokens to wait after start before enabling hooks
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
        "enabled": False,  # whether hooks should currently run
        "seen_start": False,  # have we seen start_token_id yet?
        "seen_end": False,  # have we seen end_token_id yet?
        "delay_count": 0,  # how many tokens since start we have generated
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

    gated_pre_hooks = [(m, gate_pre(h)) for (m, h) in fwd_pre_hooks]
    gated_post_hooks = [(m, gate_post(h)) for (m, h) in fwd_hooks]

    stopping = StoppingCriteriaList(
        [ToggleHooksWithDelay(start_token_id, end_token_id, gate, start_delay_n)]
    )

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
    gen_tail = generation_toks[:, tokenized.input_ids.shape[-1] :]
    logger.info(f"Number of generated tokens: {len(gen_tail[0])}")

    for i, gen in enumerate(gen_tail):
        completions.append(
            {
                "prompt": instructions[i],
                "response": tokenizer.decode(gen, skip_special_tokens=True).strip(),
            }
        )
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
        "--root_output_dir",
        type=Path,
        default=OUTPUT_ROOT,
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
        "--chosen_extraction_point",
        type=int,
        # default=EXTRACTION_POINT,
        help=(
            "The index of the extraction point corresponding to the chosen steering"
            " vector"
        ),
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
        default=list(range(0, 361, 30)),
        help="Target angles for steering",
    )
    parser.add_argument(
        "--generation_type",
        type=GenerationType,
        default=GenerationType.DEFAULT,
        help="Type of generation to use (default, early_stop or delay)",
    )
    parser.add_argument(
        "--extraction_strategy_id",
        type=str,
        default="s1_start_n_end",
        help="The strategy for choosing when to extract activations",
    )
    parser.add_argument(
        "--force_overwrite_output",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    parser.add_argument(
        "--adaptive_mode",
        type=int,
        default=0,
        help=(
            "Adaptive mode for steering (0: off, 1: adaptive to first basis, 2:"
            " adaptive to second basis"
        ),
    )
    return parser.parse_args()


@cache
def get_pytorch_infer_func(
    target_degree,
    model_id,
    root_output_dir,
    extraction_strategy_id,
    chosen_extraction_point,
    module_names,
    generation_type,
    adaptive_mode,
    upperbound_max_new_tokens,
    device,
):
    model_name = model_id.split("/")[-1]

    steering_vectors_dir = root_output_dir / "steering-vectors" / model_name
    steering_vectors_path = (
        steering_vectors_dir / extraction_strategy_id / "candidate_steering_vectors.pt"
    )

    # Load model and tokenizer with args
    model, tokenizer = load_model_and_tokenizer(model_id, device)

    tokenize_instructions_fn = functools.partial(
        tokenize_instructions_qwen_chat,
        tokenizer=tokenizer,
        enable_thinking=True,
    )

    first_basis, second_basis = calculate_basis(
        pre_computed_steering_vectors_path=Path(steering_vectors_path),
        extraction_point=chosen_extraction_point,
    )

    module_dict = dict(model.named_modules())
    module_names = module_names
    module_name_list = [
        f"model.layers.{i}.{name}"
        for i in range(model.config.num_hidden_layers)
        for name in module_names
    ]

    match generation_type:
        case GenerationType.DEFAULT:
            generate_completions_fn = generate_completions
        case GenerationType.EARLY_STOP:
            generate_completions_fn = generate_completions_early_stop
        case GenerationType.DELAY:
            generate_completions_fn = generate_completions_delay
        case _:
            raise ValueError(f"Invalid generation type: {generation_type}")

    if target_degree == "baseline":
        print("Running baseline (no steering)")
        output_hooks = []
    else:
        output_hooks = [
            (
                module_dict[module_name],
                get_angular_steering_output_hook(
                    first_direction=first_basis,
                    second_direction=second_basis,
                    target_degree=target_degree,
                    adaptive_mode=adaptive_mode,
                ),
            )
            for module_name in module_name_list
        ]

    generate_completions_fn = functools.partial(
        generate_completions_fn,
        model=model,
        tokenizer=tokenizer,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=upperbound_max_new_tokens,
        fwd_pre_hooks=[],
        fwd_hooks=output_hooks,
        system_prompt=None,
    )

    return generate_completions_fn


def main() -> None:
    # Parse command-line arguments
    args = parse_args()

    args.model_name = args.model_id.split("/")[-1]

    args.root_output_dir = Path(args.root_output_dir)

    experiment_id = f"{args.generation_type.value}-{args.extraction_strategy_id}-extraction_point_{args.chosen_extraction_point}-adaptive_{args.adaptive_mode}"

    generations_output_dir = (
        args.root_output_dir
        / "thinking-steering-generations"
        / args.model_name
        / experiment_id
    )
    generations_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    set_reproducibility(seed=args.random_seed)

    _, instructions_test = get_dataset_instructions(random_seed=args.random_seed)
    args.end_index = min(args.end_index, len(instructions_test))

    # steering_vectors_dir = args.root_output_dir / "steering-vectors" / args.model_name
    # steering_vectors_path = (
    #     steering_vectors_dir
    #     / args.extraction_strategy_id
    #     / "candidate_steering_vectors.pt"
    # )

    # # Load model and tokenizer with args
    # model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)

    # tokenize_instructions_fn = functools.partial(
    #     tokenize_instructions_qwen_chat,
    #     tokenizer=tokenizer,
    #     enable_thinking=True,
    # )

    # first_basis, second_basis = calculate_basis(
    #     pre_computed_steering_vectors_path=Path(steering_vectors_path),
    #     extraction_point=args.chosen_extraction_point,
    # )

    # module_dict = dict(model.named_modules())
    # module_names = args.module_names
    # module_name_list = [
    #     f"model.layers.{i}.{name}"
    #     for i in range(model.config.num_hidden_layers)
    #     for name in module_names
    # ]

    pprint(args.__dict__)

    print("STARTING ANGULAR STEERING EXPERIMENTS")
    print("=" * 100)

    # match args.generation_type:
    #     case GenerationType.DEFAULT:
    #         generate_completions_fn = generate_completions
    #     case GenerationType.EARLY_STOP:
    #         generate_completions_fn = generate_completions_early_stop
    #     case GenerationType.DELAY:
    #         generate_completions_fn = generate_completions_delay
    #     case _:
    #         raise ValueError(f"Invalid generation type: {args.generation_type}")

    time_per_angle = dict()

    for target_degree in args.target_angle:

        generation_output_file = generations_output_dir / f"{target_degree}.json"
        if generation_output_file.exists() and not args.force_overwrite_output:
            print(
                f"Generation output file {generation_output_file} already exists."
                " Skipping this angle."
            )
            continue

        # if target_degree == "baseline":
        #     print("Running baseline (no steering)")
        #     output_hooks = []
        # else:
        #     output_hooks = [
        #         (
        #             module_dict[module_name],
        #             get_angular_steering_output_hook(
        #                 first_direction=first_basis,
        #                 second_direction=second_basis,
        #                 target_degree=target_degree,
        #                 adaptive_mode=args.adaptive_mode,
        #             ),
        #         )
        #         for module_name in module_name_list
        #     ]

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        infer_func = get_pytorch_infer_func(
            target_degree=target_degree,
            model_id=args.model_id,
            root_output_dir=args.root_output_dir,
            extraction_strategy_id=args.extraction_strategy_id,
            chosen_extraction_point=args.chosen_extraction_point,
            module_names=tuple(args.module_names),
            generation_type=args.generation_type,
            adaptive_mode=args.adaptive_mode,
            upperbound_max_new_tokens=args.upperbound_max_new_tokens,
            device=args.device,
        )

        generation_outputs = dict()
        start_time.record()
        for i in tqdm(
            range(args.start_index, args.end_index),
            desc=f"Steering at {target_degree} degrees",
        ):
            logger.debug("-" * 100)
            logger.debug(f"QUESTION:\n{instructions_test[i]}")
            logger.debug("ANSWER:")
            # completions = generate_completions_fn(
            #     model=model,
            #     instructions=[instructions_test[i]],
            #     tokenizer=tokenizer,
            #     tokenize_instructions_fn=tokenize_instructions_fn,
            #     system_prompt=None,
            #     # system_prompt="Answer in the most angry, frustrated and aggressive way possible.",
            #     # fwd_pre_hooks=fwd_pre_hooks,
            #     fwd_hooks=output_hooks,
            #     max_new_tokens=args.upperbound_max_new_tokens,
            # )
            completions = infer_func(instructions=[instructions_test[i]])

            completions = completions[0]
            logger.debug(completions["response"][:500])
            logger.debug(
                "FIRST END THINK TOKEN POSITION:"
                f" {completions['response'].find('</think>')}"
            )
            logger.debug(
                "LAST END THINK TOKEN POSITION:"
                f" {completions['response'].rfind('</think>')}"
            )
            logger.debug("_" * 100)
            generation_outputs[i] = completions

        end_time.record()

        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time)
        time_per_angle[target_degree] = processing_time

        # # Save outputs to json at each angle
        # with open(generation_output_file, "w") as f:
        #     json.dump(generation_outputs, f, indent=4, ensure_ascii=False)
        # logger.info(
        #     f"Saved generations of {target_degree} degrees to {generation_output_file}"
        # )

    for angle, t in time_per_angle.items():
        print(f"Time taken for {angle} degrees: {t/1000:.2f} seconds")


if __name__ == "__main__":
    main()
