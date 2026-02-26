import io
from functools import cache

import pandas as pd
import requests
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import logging
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# =============================================================================
# Angular Steering Operator
# =============================================================================


def get_harmful_instructions():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train[:512], test[:128]


def get_harmful_instructions_jp():
    huggingface_id = "kunishou/do-not-answer-ja"
    dataset = load_dataset(huggingface_id)

    selected_types = [
        # # Information Hazards
        # "Risks from leaking or inferring sensitive information (organization/gov)",
        # (
        #     "Compromise privacy by leaking or inferring private information"
        #     " (person/individual)"
        # ),
        # Malicious Uses
        "Assisting illegal activities",
        "Nudging or advising users to perform unethical or unsafe actions",
        "Reducing the cost of disinformation campaigns",
        # Discrimination, Exclusion, Toxicity, Hateful, Offensive
        "Social stereotypes and unfair discrimination",
        "Toxic language (hate speech)",
        "Adult Content",
        # # Misinformation Harms
        # "Causing material harm by disseminating misinformation e.g. in medicine or law",
        # "Disseminating false or misleading information",
        # # Human-Chatbot Interaction Harms
        # "Mental Health or Overreliance Crisis",
        # "Treat Chatbot as a Human",
    ]

    instructions = []
    for item in dataset["train"]:
        if item["types_of_harm"] not in selected_types:
            continue
        instructions.append(item["question"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions_jp():
    huggingface_id = "Lazycuber/alpaca-jp"
    dataset = load_dataset(huggingface_id)

    # filter for instructions that do not have inputs
    instructions = []
    for item in dataset["train"]:
        if item["input"].strip() != "":
            continue
        inst = item["instruction"]
        inst = inst.strip("「」'")
        instructions.append(inst)

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


@cache
def get_input_data(data_type, language_id):
    if data_type == "harmless":
        if language_id == "en":
            data_train, data_test = get_harmless_instructions()
            data_train = data_train
            data_test = data_test
        elif language_id == "jp":
            data_train, data_test = get_harmless_instructions_jp()
    elif data_type == "harmful":
        if language_id == "en":
            data_train, data_test = get_harmful_instructions()
        elif language_id == "jp":
            data_train, data_test = get_harmful_instructions_jp()
    elif data_type == "sad_happy":
        emotion_prompt = "Answer in the most negative, sad and depressed way possible."
        data_train, data_test = get_harmless_instructions()
        data_train = [f"{inst}\n{emotion_prompt}" for inst in data_train]
        data_test = [f"{inst}\n{emotion_prompt}" for inst in data_test]
    elif data_type == "angry_calm":
        emotion_prompt = (
            "Answer in the most angry, frustrated and aggressive way possible."
        )
        data_train, data_test = get_harmless_instructions()
        data_train = [f"{inst}\n{emotion_prompt}" for inst in data_train]
        data_test = [f"{inst}\n{emotion_prompt}" for inst in data_test]

    return data_train, data_test


class AngularSteeringOperator:
    """
    Angular steering operator for transforming activations in a 2D plane.

    Implements the core steering transformation:
        h' = h - P*h + ||P*h|| * v_theta

    where:
        - P = b1⊗b1^T + b2⊗b2^T is the projection matrix onto the steering plane
        - v_theta = cos(θ)*b1 + sin(θ)*b2 is the target direction vector
        - θ is the rotation angle in degrees

    Supports multiple adaptive modes for conditional steering.
    """

    def __init__(self, first_direction: np.ndarray, second_direction: np.ndarray):
        """
        Initialize the steering operator with basis vectors.

        Args:
            first_direction: First basis vector (numpy array)
            second_direction: Second basis vector (numpy array)
        """
        # Convert numpy arrays to torch tensors
        self.first_direction = torch.from_numpy(first_direction).float()
        self.second_direction = torch.from_numpy(second_direction).float()

        # Precompute orthonormalized basis
        self.b1 = self.first_direction / self.first_direction.norm()
        self.b2 = self.second_direction - (self.second_direction @ self.b1) * self.b1
        self.b2 = self.b2 / self.b2.norm()

        # Precompute projection matrix: P = b1⊗b1^T + b2⊗b2^T
        self.proj_matrix = torch.outer(self.b1, self.b1) + torch.outer(self.b2, self.b2)

        # Cache for device-specific tensors
        self._device_cache: Dict[Tuple, Dict] = {}
        self._rotation_cache: Dict[Tuple, torch.Tensor] = {}

    def _get_device_tensors(self, device: torch.device, dtype: torch.dtype) -> Dict:
        """Get or create cached device-specific tensors."""
        cache_key = (device, dtype)
        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = {
                "proj_matrix": self.proj_matrix.to(device=device, dtype=dtype),
                "b1": self.b1.to(device=device, dtype=dtype),
                "b2": self.b2.to(device=device, dtype=dtype),
            }
        return self._device_cache[cache_key]

    def _get_rotation_vector(
        self, theta: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get cached rotation vector v_theta = cos(θ)*b1 + sin(θ)*b2."""
        # Normalize theta to [0, 360) for consistent caching
        theta_normalized = theta % 360
        cache_key = (device, dtype, theta_normalized)

        if cache_key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = torch.tensor(theta_normalized * torch.pi / 180.0)
            self._rotation_cache[cache_key] = (
                torch.cos(theta_rad) * cached["b1"]
                + torch.sin(theta_rad) * cached["b2"]
            )

        return self._rotation_cache[cache_key]

    def steer(
        self,
        hidden_states: torch.Tensor,
        target_degree: float,
        adaptive_mode: int = 1,
    ) -> torch.Tensor:
        """
        Apply angular steering to hidden states.

        Args:
            hidden_states: Tensor of shape (..., hidden_dim)
            target_degree: Rotation angle in degrees (0-360)
            adaptive_mode: Steering application mode:
                0 = Always steer all activations (non-adaptive)
                1 = Only steer when activation is aligned with first_direction
                    (conditional steering based on positive projection)

        Returns:
            Steered hidden states with same shape as input
        """
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get cached tensors for this device/dtype
        cached = self._get_device_tensors(device, dtype)
        proj_matrix = cached["proj_matrix"]
        first_dir = cached["b1"]

        # Get rotation vector
        v_theta = self._get_rotation_vector(target_degree, device, dtype)

        # Project onto steering plane: proj_h = h @ P^T
        proj_h = hidden_states @ proj_matrix.T

        # Compute magnitude: r = ||P*h||
        r = proj_h.norm(dim=-1, keepdim=True)

        if adaptive_mode == 0:
            # Non-adaptive: always steer
            # h' = h - P*h + r * v_theta
            steered = hidden_states - proj_h + r * v_theta
            return steered

        elif adaptive_mode == 1:
            # Adaptive: only steer when aligned with harmful direction
            # Compute alignment with first direction
            alignment = hidden_states @ first_dir

            # Create mask: steer only when alignment > 0
            mask = (alignment > 0).unsqueeze(-1)

            # h' = h + mask * (r * v_theta - P*h)
            steered = hidden_states - proj_h + r * v_theta
            return torch.where(mask, steered, hidden_states)

        else:
            raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}. Supported: 0, 1")

    def clear_cache(self):
        """Clear all cached tensors."""
        self._device_cache.clear()
        self._rotation_cache.clear()

    def clear_rotation_cache(self):
        """Clear only rotation cache (preserves device tensors)."""
        self._rotation_cache.clear()


# =============================================================================
# Hook Creation and Management
# =============================================================================


def _detect_prefill_decode_phase(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> bool:
    """
    Detect if we're in decode phase using vLLM's attention metadata.

    Uses vLLM's internal forward_context to check max_query_len:
    - Prefill: max_query_len > 1 (processing multiple input tokens)
    - Decode: max_query_len == 1 (generating one token at a time)

    Args:
        hidden_states: Current hidden states tensor
        layer_name: Name of the current layer (for logging)

    Returns:
        True if in decode phase, False if in prefill phase
    """
    try:
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        attn_metadata = forward_ctx.attn_metadata

        # For v1 engine, attn_metadata might be a dict or direct metadata
        if isinstance(attn_metadata, dict):
            # Get metadata from first available layer
            attn_meta = next(iter(attn_metadata.values())) if attn_metadata else None
        else:
            attn_meta = attn_metadata

        if attn_meta is not None:
            # Check max_query_len - most authoritative indicator
            max_query_len = getattr(attn_meta, "max_query_len", None)
            if max_query_len is not None:
                return max_query_len == 1  # 1 = decode, >1 = prefill

            # Fallback: Check num_decode_tokens
            if hasattr(attn_meta, "num_decode_tokens"):
                return attn_meta.num_decode_tokens > 0

            # Fallback: Check num_prefill_tokens
            if hasattr(attn_meta, "num_prefill_tokens"):
                return attn_meta.num_prefill_tokens == 0

    except Exception as e:
        logger.debug(f"Metadata detection failed for {layer_name}: {e}")

    # If metadata detection fails, assume prefill (safer default)
    return False


def clear_hooks(model: nn.Module) -> int:
    """
    Clear all forward hooks from a model.

    Args:
        model: PyTorch model

    Returns:
        Number of hooks cleared
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "_forward_hooks") and module._forward_hooks:
            count += len(module._forward_hooks)
            module._forward_hooks.clear()
    return count


def create_steering_hook(
    operator: AngularSteeringOperator,
    state: Dict,
    layer_name: str,
    prompt_only: bool = False,
) -> callable:
    """
    Create a forward hook for angular steering.

    Args:
        operator: Shared steering operator instance
        state: Mutable dict containing 'target_degree', 'adaptive_mode', 'enabled'
        layer_name: Name of the layer this hook is attached to
        prompt_only: If True, only steer prompt (not generation). If False, steer all tokens.

    Returns:
        Hook function that applies steering
    """
    _layer_name = layer_name
    _initial_operator = operator

    def hook_fn(module, input_tuple, output):
        import builtins

        # Read mutable state
        target_degree = state.get("target_degree", 0.0)
        adaptive_mode = state.get("adaptive_mode", 1)
        enabled = state.get("enabled", True)

        if not enabled:
            return output

        # Handle tuple outputs for forward compatibility
        # - LayerNorm/RMSNorm: Always return single tensor (current use case)
        # - Attention modules: Return (attn_output, attn_weights) tuple (future use case)
        # - MLP modules: Return single tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Prompt-only mode: skip steering during decode phase
        if prompt_only:
            is_decode = _detect_prefill_decode_phase(hidden_states, _layer_name)
            if is_decode:
                return output

        # Get current operator (supports dynamic updates)
        current_operator = getattr(builtins, "_steering_operator", _initial_operator)

        # Clear rotation cache when theta changes to prevent OOM
        last_theta = state.get("last_theta", None)
        if last_theta is not None and last_theta != target_degree:
            current_operator.clear_rotation_cache()
        state["last_theta"] = target_degree

        # Apply steering
        steered = current_operator.steer(
            hidden_states=hidden_states,
            target_degree=target_degree,
            adaptive_mode=adaptive_mode,
        )

        # Return in same format as input (preserve tuple structure if present)
        if rest is not None:
            return (steered,) + rest
        return steered

    return hook_fn


def register_hooks_fn(
    model: nn.Module,
    shared_operator,
    target_layers,
    target_degree,
    adaptive_mode,
    prompt_only,
):
    """Register hooks on target layers in worker process."""
    import builtins

    # Create shared mutable state in worker process
    if not hasattr(builtins, "_steering_state"):
        builtins._steering_state = {}

    builtins._steering_state["target_degree"] = target_degree
    builtins._steering_state["adaptive_mode"] = adaptive_mode
    builtins._steering_state["enabled"] = True

    builtins._steering_state["is_first_pass"] = True
    builtins._steering_state["last_theta"] = None

    # Store operator reference
    builtins._steering_operator = shared_operator

    # Remove existing hooks
    clear_hooks(model)

    count = 0
    hooked_layers = []

    # Get module dict
    module_dict = dict(model.named_modules())

    for layer_name in target_layers:
        if layer_name in module_dict:
            module = module_dict[layer_name]

            # Create hook with shared operator and state
            hook = create_steering_hook(
                operator=shared_operator,
                state=builtins._steering_state,
                layer_name=layer_name,
                prompt_only=prompt_only,
            )

            # Register hook
            module.register_forward_hook(hook)
            count += 1
            hooked_layers.append(layer_name)

    return count


def update_state_fn(model: nn.Module, target_degree, adaptive_mode, enabled):
    import builtins

    if hasattr(builtins, "_steering_state"):
        if target_degree is not None:
            builtins._steering_state["target_degree"] = target_degree
        if adaptive_mode is not None:
            builtins._steering_state["adaptive_mode"] = adaptive_mode
        if enabled is not None:
            builtins._steering_state["enabled"] = enabled
    return True


def remove_hooks_fn(model: nn.Module):
    return clear_hooks(model)
