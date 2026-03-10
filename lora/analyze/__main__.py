"""CLI entry point: python -m lora.analyze

Usage:
    python -m lora.analyze --config lora/configs/example_qwen3b.yaml
    python -m lora.analyze --config ... --skip-activation
    python -m lora.analyze --config ... --directions-file path/to/file.npy
    python -m lora.analyze --config ... --n-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# ── Model-ID → directions-file stem map ───────────────────────────────────────
# Avoids importing root configs.py which has heavy dependencies.
# Format: model_id → "{strategy}_{layer}_{position}"
_DIR_IDS: dict[str, str] = {
    "Qwen/Qwen2.5-3B-Instruct": "max_sim_25_mid",
    "Qwen/Qwen2.5-7B-Instruct": "max_sim_25_mid",
    "meta-llama/Llama-3.1-8B-Instruct": "max_sim_22_mid",
    "meta-llama/Llama-3.2-3B-Instruct": "max_sim_22_mid",
}

# Repo root = analyze/ → lora/ → repo root
_REPO_ROOT = Path(__file__).parent.parent.parent

# Fallback search path for directions files (pytorch_pure outputs first)
_DIR_SEARCH_ROOTS = [
    _REPO_ROOT / "pytorch_pure" / "output",
    _REPO_ROOT / "output",
]


def _find_directions_file(model_id: str) -> Optional[Path]:
    """Auto-derive the directions file path from the model_id."""
    dir_id = _DIR_IDS.get(model_id)
    if not dir_id:
        return None

    model_short = model_id.split("/")[-1]

    # Try both naming conventions: with and without "dir_" prefix
    candidates = [
        f"steering_config-en-dir_{dir_id}-pca_0.npy",
        f"steering_config-en-{dir_id}-pca_0.npy",
    ]

    for root in _DIR_SEARCH_ROOTS:
        for fname in candidates:
            p = root / model_short / fname
            if p.exists():
                return p

    return None


def _build_summary(weight_rows: list, activation_rows: Optional[list]) -> str:
    lines = []

    # ── Weight analysis table ──────────────────────────────────────────────────
    lines.append(
        "Weight-Space Analysis (ΔW = B·A SVD, cos-sim with angular steering first_direction)"
    )
    lines.append(f"{'Layer':>5}  {'Module':<8}  {'‖ΔW‖_F':>8}  {'S[0]':>8}  {'cos_input':>10}  {'cos_output':>10}")
    lines.append("-" * 60)

    best_steering_layer = 25  # known best layer
    for row in weight_rows:
        L = row["layer_idx"]
        mod = row["module"]
        frob = row["frobenius_norm"]
        s0 = row["singular_values"][0] if row["singular_values"] else float("nan")
        ci = row["cos_sim_input"]
        co = row["cos_sim_output"]

        ci_str = f"{ci:+.3f}" if ci is not None else "  null"
        co_str = f"{co:+.3f}" if co is not None else "   n/a"

        marker = "  ← best steering layer" if L == best_steering_layer and mod == "q_proj" else ""
        lines.append(
            f"{L:5d}  {mod:<8}  {frob:8.4f}  {s0:8.4f}  {ci_str:>10}  {co_str:>10}{marker}"
        )

    lines.append("")

    # ── Activation analysis table ──────────────────────────────────────────────
    if activation_rows:
        lines.append(
            "Activation-Delta Analysis (h_lora − h_base vs angular steering first_direction)"
        )
        lines.append(f"{'Layer':>5}  {'‖Δh‖':>10}  {'cos(Δh,d1)':>12}")
        lines.append("-" * 35)

        max_delta_layer = max(activation_rows, key=lambda r: r["delta_norm"])["layer_idx"]

        for row in activation_rows:
            L = row["layer_idx"]
            dn = row["delta_norm"]
            cs = row["cos_sim_first_direction"]
            cs_str = f"{cs:+.3f}" if cs is not None else "  null"
            marker = "  ← max delta" if L == max_delta_layer else ""
            lines.append(f"{L:5d}  {dn:10.5f}  {cs_str:>12}{marker}")

        lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    lines.append("Key findings:")

    # Max |cos_input| in weight analysis
    valid_ci = [(r["layer_idx"], r["module"], r["cos_sim_input"])
                for r in weight_rows if r["cos_sim_input"] is not None]
    if valid_ci:
        L, mod, cos = max(valid_ci, key=lambda x: abs(x[2]))
        lines.append(f"  Max |cos_input|  in weight analysis: layer={L}, module={mod}, cos={cos:+.4f}")

    if activation_rows:
        # Max ‖Δh‖
        max_dn = max(activation_rows, key=lambda r: r["delta_norm"])
        lines.append(f"  Max ‖Δh‖         in activation analysis: layer={max_dn['layer_idx']}")

        # Max |cos(Δh,d1)|
        valid_cs = [(r["layer_idx"], r["cos_sim_first_direction"])
                    for r in activation_rows if r["cos_sim_first_direction"] is not None]
        if valid_cs:
            L, cos = max(valid_cs, key=lambda x: abs(x[1]))
            lines.append(f"  Max |cos(Δh,d1)| in activation analysis: layer={L}, cos={cos:+.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse LoRA weights vs angular steering directions"
    )
    parser.add_argument("--config", required=True, help="Path to YAML LoRA config")
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Path to LoRA weights directory (default: config.run_dir/lora_weights)",
    )
    parser.add_argument(
        "--directions-file",
        default=None,
        help="Path to .npy steering directions file (default: auto-derived from model_id)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of prompts for activation analysis (default: 20)",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="Skip activation analysis (weight analysis only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation forward passes (default: 4)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config ────────────────────────────────────────────────────────────
    from ..config import LoRAConfig
    config = LoRAConfig.from_yaml(args.config)

    # ── Resolve LoRA path ──────────────────────────────────────────────────────
    lora_path = Path(args.lora_path) if args.lora_path else config.run_dir / "lora_weights"
    if not lora_path.exists():
        print(f"Error: LoRA path not found: {lora_path}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve directions file ────────────────────────────────────────────────
    if args.directions_file:
        directions_file = Path(args.directions_file)
    else:
        directions_file = _find_directions_file(config.model_id)
        if directions_file is None:
            print(
                f"Error: Could not auto-derive directions file for model '{config.model_id}'.\n"
                f"  Pass --directions-file explicitly.\n"
                f"  Searched roots: {[str(r) for r in _DIR_SEARCH_ROOTS]}\n"
                f"  Known model IDs: {list(_DIR_IDS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not directions_file.exists():
        print(
            f"Error: Directions file not found: {directions_file}\n"
            f"  If this is a git-LFS pointer, pull it:\n"
            f"  git lfs pull --include \"{directions_file}\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Output directory ───────────────────────────────────────────────────────
    out_dir = config.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:          {args.config}")
    print(f"LoRA path:       {lora_path}")
    print(f"Directions file: {directions_file}")
    print(f"Output dir:      {out_dir}")
    print()

    # ── Weight analysis ────────────────────────────────────────────────────────
    from .weight_analysis import run_weight_analysis

    print("Running weight analysis …")
    weight_rows = run_weight_analysis(
        lora_path=lora_path,
        directions_file=directions_file,
        lora_rank=config.lora_rank,
    )

    weight_path = out_dir / "weight_analysis.json"
    weight_path.write_text(json.dumps(weight_rows, indent=2))
    print(f"  Saved: {weight_path}  ({len(weight_rows)} rows)")

    # ── Activation analysis ────────────────────────────────────────────────────
    activation_rows = None
    if not args.skip_activation:
        from .activation_analysis import run_activation_analysis

        print(f"Running activation analysis (n_samples={args.n_samples}) …")
        activation_rows = run_activation_analysis(
            config=config,
            lora_path=lora_path,
            directions_file=directions_file,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
        )
        act_path = out_dir / "activation_analysis.json"
        act_path.write_text(json.dumps(activation_rows, indent=2))
        print(f"  Saved: {act_path}  ({len(activation_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = _build_summary(weight_rows, activation_rows)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print(f"  Saved: {summary_path}")

    print()
    print(summary)


if __name__ == "__main__":
    main()
