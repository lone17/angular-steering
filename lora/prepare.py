"""Prepare LoRA training data using vLLM angular steering (--stage prepare).

Called from pipeline.py; not intended to be run directly.
"""

import json
import logging
import os
from pathlib import Path

from .config import PrepareConfig
from .data import _is_refusal


def prepare_data(config: PrepareConfig, logger: logging.Logger) -> dict:
    """Generate steered harmful + harmless-baseline training data.

    Writes output files into config.resolved_data_dir and returns a dict
    with paths to the generated files (for inline use with --stage all).

    Uses data_n_harmless to control whether harmless data is generated:
    - data_n_harmless > 0: generate specific number
    - data_n_harmless == 0: skip harmless data
    - data_n_harmless == -1: all available

    Args:
        config: Prepare-stage config. Must have steering_config_file set.
        logger: Logger to write progress messages to.

    Returns:
        Dict with keys: harmful_prompts_file, harmful_responses_file,
        harmless_prompts_file (if generated), harmless_responses_file (if generated).
    """
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from pytorch_pure.utils import get_input_data  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm_angular_steering import (  # noqa: PLC0415
        AngularSteering,
        _format_prompts_for_vllm,
    )

    if not config.steering_config_file:
        raise ValueError(
            "steering_config_file must be set in the config for --stage prepare"
        )

    config_file = Path(config.steering_config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Steering config file not found: {config_file}")

    angles = config.data_angles if config.data_angles else [0, 90, 180]
    split_tag = config.data_split
    adaptive_mode = config.data_adaptive_mode

    # Derive naming stem from config file (same convention as the old generate_data.py)
    stem_parts = config_file.stem.split("-")
    dir_stem = "-".join(stem_parts[2:]) if len(stem_parts) > 2 else config_file.stem
    adaptive_label = f"adaptive_{adaptive_mode}"

    data_dir = config.resolved_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    harmful_prompts_path = data_dir / f"harmful-en-{split_tag}-samples.json"
    harmful_responses_path = (
        data_dir / f"harmful-en-{split_tag}-{dir_stem}-{adaptive_label}.json"
    )
    harmless_prompts_path = data_dir / f"harmless-en-{split_tag}-samples.json"
    harmless_responses_path = data_dir / f"harmless-en-{split_tag}-baseline.json"

    def _save_json(path: Path, obj) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        logger.info(f"  Saved → {path}")

    # ── Load prompts ────────────────────────────────────────────────────────────
    logger.info("Loading AdvBench harmful instructions …")
    harm_train, harm_test = get_input_data("harmful", "en")
    harmful_prompts = harm_train if split_tag == "train" else harm_test
    if config.data_n_harmful > 0:
        harmful_prompts = harmful_prompts[: config.data_n_harmful]

    harmless_prompts: list[str] = []
    generate_harmless = config.data_n_harmless != 0  # 0 means skip, -1 means all

    if generate_harmless:
        logger.info("Loading Alpaca harmless instructions …")
        less_train, less_test = get_input_data("harmless", "en")
        harmless_prompts = less_train if split_tag == "train" else less_test
        if config.data_n_harmless > 0:
            harmless_prompts = harmless_prompts[: config.data_n_harmless]

    logger.info(
        f"  harmful={len(harmful_prompts)}, harmless={len(harmless_prompts)}"
    )

    if not harmful_prompts_path.exists():
        _save_json(harmful_prompts_path, harmful_prompts)
    else:
        logger.info(f"  Harmful prompts already exist: {harmful_prompts_path}")

    if generate_harmless and not harmless_prompts_path.exists():
        _save_json(harmless_prompts_path, harmless_prompts)
    elif generate_harmless:
        logger.info(f"  Harmless prompts already exist: {harmless_prompts_path}")

    # ── Initialise vLLM ────────────────────────────────────────────────────────
    # Hardcoded sensible defaults for data generation
    gpu_memory_utilization = 0.9
    tensor_parallel_size = 1
    max_tokens = 512

    logger.info(f"Initialising vLLM: {config.model_id}")
    llm = LLM(
        model=config.model_id,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # ── Step 1: Steered responses for harmful prompts ────────────────────────────
    if not harmful_responses_path.exists():
        logger.info(f"Steered harmful responses — angles: {angles}")
        steering = AngularSteering(llm)
        steering.load_config_from_file(str(config_file))
        steering.apply_steering(target_degree=angles[0], adaptive_mode=adaptive_mode)

        steered: dict[str, list[str]] = {}
        chat_harmful = _format_prompts_for_vllm(harmful_prompts)
        for angle in angles:
            logger.info(f"  angle={angle}° …")
            steering.set_degree(angle)
            outputs = llm.chat(chat_harmful, sampling_params=params)
            steered[str(angle)] = [o.outputs[0].text for o in outputs]

        steering.remove_steering()

        # ── Filter refusals from steered responses ───────────────────────────
        if config.filter_refusals:
            primary_key = str(angles[0])
            n_before = len(harmful_prompts)
            keep = [
                i for i, r in enumerate(steered.get(primary_key, []))
                if not _is_refusal(r)
            ]
            harmful_prompts = [harmful_prompts[i] for i in keep]
            for k in steered:
                steered[k] = [steered[k][i] for i in keep]
            n_dropped = n_before - len(keep)
            logger.info(
                f"  filter_refusals (angle={angles[0]}°): "
                f"{len(keep)}/{n_before} kept, {n_dropped} refusals dropped"
            )
            # Re-save filtered prompts (overwrite the file saved above)
            _save_json(harmful_prompts_path, harmful_prompts)

        _save_json(harmful_responses_path, steered)
    else:
        logger.info(f"Harmful responses already exist: {harmful_responses_path}")

    # ── Step 2: Baseline responses for harmless prompts ──────────────────────────
    if generate_harmless and not harmless_responses_path.exists():
        logger.info("Harmless baseline responses (no steering) …")
        baseline_outputs = llm.chat(
            _format_prompts_for_vllm(harmless_prompts), sampling_params=params
        )
        _save_json(
            harmless_responses_path,
            {"baseline": [o.outputs[0].text for o in baseline_outputs]},
        )
    elif generate_harmless:
        logger.info(f"Harmless responses already exist: {harmless_responses_path}")

    # ── Return paths for inline use (--stage all) ─────────────────────────────
    result = {
        "harmful_prompts_file": str(harmful_prompts_path),
        "harmful_responses_file": str(harmful_responses_path),
    }
    if generate_harmless:
        result["harmless_prompts_file"] = str(harmless_prompts_path)
        result["harmless_responses_file"] = str(harmless_responses_path)

    logger.info(f"Data ready → {data_dir}")
    logger.info(f"  harmful_prompts_file: {result['harmful_prompts_file']}")
    logger.info(f"  harmful_responses_file: {result['harmful_responses_file']}")
    if generate_harmless:
        logger.info(f"  harmless_prompts_file: {result['harmless_prompts_file']}")
        logger.info(f"  harmless_responses_file: {result['harmless_responses_file']}")

    return result
