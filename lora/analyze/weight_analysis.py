"""Weight-space analysis: SVD of LoRA delta weights vs angular steering directions."""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_LFS_POINTER_MARKER = "version https://git-lfs.github.com/spec/v1"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_directions(directions_file: str | Path) -> dict:
    """Load steering directions from a .npy file (dict with allow_pickle).

    Raises:
        ValueError: If the file is a git-LFS pointer.
    """
    path = Path(directions_file)
    # Detect LFS pointer
    try:
        with open(path, "r") as f:
            first_line = f.readline().strip()
        if first_line == _LFS_POINTER_MARKER:
            raise ValueError(
                f"File '{path}' is a git-LFS pointer. Pull it first:\n"
                f"  git lfs pull --include \"{path}\""
            )
    except UnicodeDecodeError:
        pass  # Binary file — not a pointer

    raw = np.load(path, allow_pickle=True)
    return raw.item() if raw.ndim == 0 else dict(raw)


def load_lora_weights(lora_path: str | Path) -> dict:
    """Parse adapter_model.safetensors into {layer_idx: {module: {"A": arr, "B": arr}}}.

    Args:
        lora_path: Path to the directory containing adapter_model.safetensors,
                   OR path directly to the .safetensors file.
    """
    from safetensors.torch import safe_open

    lora_path = Path(lora_path)
    if lora_path.is_dir():
        lora_path = lora_path / "adapter_model.safetensors"

    pattern = re.compile(
        r"layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.weight"
    )

    result: dict[int, dict[str, dict]] = {}

    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            m = pattern.search(key)
            if not m:
                continue
            layer_idx = int(m.group(1))
            module = m.group(2)
            ab = m.group(3)

            tensor = f.get_tensor(key)
            # Cast bfloat16 → float32 before numpy
            arr = tensor.to(dtype=tensor.float().dtype).numpy()

            result.setdefault(layer_idx, {}).setdefault(module, {})
            result[layer_idx][module][ab] = arr

    logger.info(
        f"Loaded LoRA weights: {len(result)} layers, "
        f"modules={sorted({m for l in result.values() for m in l})}"
    )
    return result


def run_weight_analysis(
    lora_path: str | Path,
    directions_file: str | Path,
    lora_rank: int = 4,
) -> list[dict]:
    """Compute per-layer, per-module SVD of ΔW and alignment with steering directions.

    For each (layer L, module M ∈ {q_proj, v_proj}):
      - ΔW = B @ A
      - SVD → U, S, Vt
      - cos_sim_input:  cosine(Vt[0,:], first_direction from input_layernorm[L])
      - cos_sim_output: cosine(U[:,0],  first_direction from post_attention_layernorm[L])
                        (null for v_proj whose output dim ≠ hidden_size)

    Returns:
        List of dicts, one per (layer, module).
    """
    directions = load_directions(directions_file)
    weights = load_lora_weights(lora_path)

    rows = []
    for layer_idx in sorted(weights):
        layer_mods = weights[layer_idx]
        for module in sorted(layer_mods):
            ab = layer_mods[module]
            if "A" not in ab or "B" not in ab:
                logger.warning(f"Layer {layer_idx} {module}: missing A or B, skipping")
                continue

            A = ab["A"]  # [rank, hidden]
            B = ab["B"]  # [out, rank]
            delta_W = B @ A  # [out, hidden]

            U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
            frob = float(np.linalg.norm(S))

            # --- Input-space cosine (Vt[0,:] vs first_direction @ input_layernorm) ---
            in_key = f"model.layers.{layer_idx}.input_layernorm"
            cos_sim_input: Optional[float]
            if in_key in directions:
                fd_input = directions[in_key]["first_direction"]
                cos_sim_input = _cosine(Vt[0, :], fd_input)
            else:
                logger.debug(f"  {in_key} not in directions; cos_sim_input=null")
                cos_sim_input = None

            # --- Output-space cosine (U[:,0] vs first_direction @ post_attention_layernorm) ---
            out_key = f"model.layers.{layer_idx}.post_attention_layernorm"
            cos_sim_output: Optional[float]
            hidden_size = A.shape[1]
            if module == "v_proj" and B.shape[0] != hidden_size:
                # v_proj output dim ≠ hidden_size → can't compare
                cos_sim_output = None
            elif out_key in directions:
                fd_output = directions[out_key]["first_direction"]
                cos_sim_output = _cosine(U[:, 0], fd_output)
            else:
                logger.debug(f"  {out_key} not in directions; cos_sim_output=null")
                cos_sim_output = None

            row = {
                "layer_idx": layer_idx,
                "module": module,
                "frobenius_norm": frob,
                "singular_values": S[:lora_rank].tolist(),
                "cos_sim_input": cos_sim_input,
                "cos_sim_output": cos_sim_output,
            }
            rows.append(row)
            logger.debug(
                f"  L{layer_idx:2d} {module:6s}  ‖ΔW‖={frob:.4f}  "
                f"cos_in={_fmt(cos_sim_input)}  cos_out={_fmt(cos_sim_output)}"
            )

    return rows


def _fmt(v: Optional[float]) -> str:
    return f"{v:+.3f}" if v is not None else " null"
