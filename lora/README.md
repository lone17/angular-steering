# lora/

Trains a LoRA adapter to approximate angular steering at a fixed angle (default: 180° = maximum refusal), and provides tools to analyse how the resulting weights relate to the original steering directions.

---

## Concept

[Angular steering](../pytorch_pure/) modifies model activations at inference time to steer the model toward or away from a target direction. This module asks: *can a LoRA adapter achieve the same behaviour statically, by baking the steering into the weights?*

Three training objectives are supported:

| Objective | Description |
|---|---|
| `sft` | Standard SFT on `(harmful prompt → steered refusal)` pairs |
| `sft_combined` | SFT on harmful pairs + harmless baseline pairs pooled — prevents always-refuse collapse |
| `dpo` | Direct Preference Optimisation: chosen = steered refusal, rejected = harmful compliance |

The `analyze/` sub-module compares the resulting LoRA weight directions against the original steering directions.

---

## Directory layout

```
lora/
    config.py               — PrepareConfig + LoRAConfig dataclasses (YAML-serialisable)
    model.py                — load_base_model(), add_lora(), load_lora_model()
    data.py                 — SFTDataset, DPODataset, load_steered_pairs(), …
    prepare.py              — prepare_data() — generate training data (--stage prepare)
    trainer.py              — train() — dispatches to SFT / SFT-combined / DPO
    evaluate.py             — evaluate() — HF generate + side-by-side comparison
    evaluate_vllm.py        — evaluate_with_vllm() — vLLM LoRARequest backend
    pipeline.py             — CLI entry point wiring prepare → train → eval
    __main__.py             — python -m lora

    analyze/
        __init__.py         — run_analysis() convenience wrapper
        weight_analysis.py  — SVD of ΔW per layer, cosine-sim with steering directions
        activation_analysis.py — h_lora − h_base deltas per layer
        __main__.py         — python -m lora.analyze

    configs/
        prepare_qwen3b.yaml                 — PrepareConfig: SFT data (harmful only)
        train_qwen3b.yaml                   — LoRAConfig: SFT training on prepared data
        prepare_qwen3b_sft_combined.yaml    — PrepareConfig: SFT-combined data (harmful + harmless)
        train_qwen3b_sft_combined.yaml      — LoRAConfig: SFT-combined training
        prepare_qwen3b_dpo.yaml             — PrepareConfig: DPO data (harmful chosen)
        train_qwen3b_dpo.yaml               — LoRAConfig: DPO training

    output/                 — per-run directories + intermediate data (gitignored)
    logs/                   — per-run log files (gitignored)
```

---

## Quick start

All commands run from the **repo root** (`angular-steering/`).

### 0 — Prepare training data

Two-stage workflow: data preparation and training use **separate config files**.

Generate steered harmful responses (and optionally harmless baseline responses) and write them to the intermediate data directory so you can inspect them before training:

```bash
python -m lora --config lora/configs/prepare_qwen3b.yaml --stage prepare
```

Outputs are written to `data_dir` (default: `output/{model_short}/`):

| File | Content |
|---|---|
| `harmful-en-{split}-samples.json` | Harmful prompt list (AdvBench) |
| `harmful-en-{split}-{stem}-adaptive_1.json` | Steered responses `{"180": [response, …], …}` |
| `harmless-en-{split}-samples.json` | Harmless prompt list (Alpaca) — SFT-combined only |
| `harmless-en-{split}-baseline.json` | Unsteered baseline responses — SFT-combined only |

Inspect the JSON files, then proceed to training when satisfied.

### 1 — Train

Use the corresponding training config that points to the prepared data from step 0:

```bash
# SFT (harmful steered pairs only) — train + eval
python -m lora --config lora/configs/train_qwen3b.yaml --stage all

# SFT-combined (harmful + harmless pooled)
python -m lora --config lora/configs/train_qwen3b_sft_combined.yaml --stage all

# DPO
python -m lora --config lora/configs/train_qwen3b_dpo.yaml --stage all
```

Train only (skip eval):

```bash
python -m lora --config lora/configs/train_qwen3b.yaml --stage train
```

Hyperparameter overrides (no YAML edit needed):

```bash
python -m lora --config lora/configs/train_qwen3b.yaml --rank 8
python -m lora --config lora/configs/train_qwen3b.yaml --rank 1 --angle 90 --epochs 5
python -m lora --config lora/configs/train_qwen3b.yaml --modules q_proj,k_proj,v_proj,o_proj
```

### 2 — Evaluate

```bash
# HuggingFace backend (default)
python -m lora --config lora/configs/train_qwen3b.yaml --stage eval

# vLLM backend (faster; uses LoRARequest — no model reload)
python -m lora --config lora/configs/train_qwen3b.yaml --stage eval --backend vllm
```

### 3 — Analyse

Weight-space analysis only (no GPU needed beyond loading weights):

```bash
python -m lora.analyze --config lora/configs/train_qwen3b.yaml --skip-activation
```

Full analysis (weight + activation deltas):

```bash
python -m lora.analyze --config lora/configs/train_qwen3b.yaml --n-samples 20
```

Override paths:

```bash
python -m lora.analyze \
    --config lora/configs/train_qwen3b.yaml \
    --lora-path /path/to/lora_weights \
    --directions-file pytorch_pure/output/Qwen2.5-3B-Instruct/steering_config-en-max_sim_25_mid-pca_0.npy
```

---

## Config

Two separate YAML config types:

### PrepareConfig (`--stage prepare`)

Used only for the data generation stage. Specifies which angles to generate, how many samples, and where to write the output.

**Key fields:**

| Field | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `steering_config_file` | — | Path to `.npy` steering directions file (from `pytorch_pure/`) |
| `data_dir` | `output/{model_short}/` | Directory where intermediate training data is written |
| `data_split` | `"train"` | AdvBench / Alpaca split (`"train"` / `"test"`) |
| `data_angles` | `[]` | Steering angles to generate responses at |
| `data_n_harmful` | `-1` | Max harmful prompts; `-1` = all |
| `data_n_harmless` | `-1` | Max harmless prompts; `-1` = all, `0` = skip |
| `data_adaptive_mode` | `1` | Steering mode: `1` = conditional, `0` = unconditional |

**Hardcoded defaults in code** (not configurable):
- `gpu_memory_utilization = 0.9` (vLLM)
- `tensor_parallel_size = 1`
- `max_tokens = 512`

### LoRAConfig (`--stage train`, `--stage eval`, `--stage all`)

Used for training and evaluation. Points to pre-generated training data from the prepare stage, specifies the training objective, and controls LoRA hyperparameters.

**Core fields:**

| Field | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `training_objective` | `sft` | `"sft"` / `"sft_combined"` / `"dpo"` |
| `harmful_prompts_file` | — | JSON list of harmful prompt strings |
| `harmful_responses_file` | — | JSON dict `{angle_str: [response, …]}` from prepare stage |
| `steering_angle` | `180` | Which angle's responses to use as training target |
| `filter_refusals` | `true` | Drop responses without a refusal marker |
| `pool_all_angles` | `false` | Collect refusals from every angle in the responses file |
| `n_train` | `-1` | Max harmful training samples (`-1` = all) |

**SFT-combined extras** (`training_objective: sft_combined`):

| Field | Default | Description |
|---|---|---|
| `harmless_prompts_file` | — | JSON list of harmless prompt strings |
| `harmless_responses_file` | — | JSON dict `{key: [response, …]}` from prepare stage |
| `harmless_angle` | `"baseline"` | Key to read from harmless responses dict |
| `n_harmless` | `-1` | Max harmless samples (`-1` = all) |

**DPO extras** (`training_objective: dpo`):

| Field | Default | Description |
|---|---|---|
| `dpo_rejected_file` | — | Baseline/harmful responses for the rejected side |
| `dpo_rejected_angle` | `null` | Key in rejected file (`null` = first key or flat list) |
| `dpo_beta` | `0.1` | KL-regularisation coefficient β |

**LoRA:**

| Field | Default | Description |
|---|---|---|
| `lora_rank` | `4` | LoRA rank (r) |
| `lora_alpha` | `16.0` | LoRA scaling = alpha / rank |
| `lora_dropout` | `0.05` | LoRA dropout |
| `lora_target_modules` | `[q_proj, v_proj]` | Attention modules to adapt |

**Training:**

| Field | Default | Description |
|---|---|---|
| `learning_rate` | `2e-4` | AdamW learning rate |
| `num_epochs` | `3` | Training epochs |
| `per_device_batch_size` | `2` | Batch size per GPU |
| `gradient_accumulation_steps` | `4` | Gradient accumulation |
| `max_seq_length` | `512` | Max token length (truncates longer examples) |
| `warmup_ratio` | `0.05` | Warmup fraction of total steps |
| `weight_decay` | `0.01` | AdamW weight decay |
| `bf16` | `true` | Use bfloat16 mixed precision |

**Output & Logging:**

| Field | Default | Description |
|---|---|---|
| `output_dir` | `output/` | Base output directory |
| `run_name` | auto | Override auto-generated run identifier |
| `n_eval` | `20` | Number of samples to evaluate |
| `eval_max_new_tokens` | `256` | Max tokens during eval generation |
| `eval_batch_size` | `4` | Batch size for eval |

The auto-generated `run_name` encodes key parameters:

```
Qwen2.5-3B-Instruct__rank4__angle180__mods-q+v__data-<dataset-stem>__obj-sft_combined
```

---

## Workflow example

Complete step-by-step example using SFT-combined training:

```bash
# Step 1: Generate both harmful and harmless training data
python -m lora --config lora/configs/prepare_qwen3b_sft_combined.yaml --stage prepare

# Step 2: Inspect the generated JSON files in output/Qwen2.5-3B-Instruct/
ls -la output/Qwen2.5-3B-Instruct/
# should show: harmful-en-train-*.json, harmless-en-train-*.json, etc.

# Step 3: Train on the prepared data
python -m lora --config lora/configs/train_qwen3b_sft_combined.yaml --stage train

# Step 4: Evaluate the trained adapter
python -m lora --config lora/configs/train_qwen3b_sft_combined.yaml --stage eval

# Or do steps 3-4 together with --stage all
python -m lora --config lora/configs/train_qwen3b_sft_combined.yaml --stage all

# Step 5: Analyze the adapter weights
python -m lora.analyze --config lora/configs/train_qwen3b_sft_combined.yaml --n-samples 50
```

---

## Output structure

Each training run creates:

```
{output_dir}/{run_name}/
    config.yaml             — frozen config snapshot
    train_metrics.json      — loss, runtime, samples/sec
    eval_results.json       — side-by-side (prompt, lora, steered, baseline)
    eval_results_vllm.json  — same, vLLM backend
    lora_weights/           — PEFT adapter (adapter_config.json + safetensors)
    checkpoints/            — HF Trainer checkpoints (last epoch)
    analysis/               — output of python -m lora.analyze
        weight_analysis.json
        activation_analysis.json
        summary.txt
```

Intermediate training data from `--stage prepare` is written to `data_dir` (e.g., `output/{model_short}/`), shared across multiple training runs.

---

## Analysis module

`python -m lora.analyze` compares the trained adapter to the angular steering directions in two ways.

### Weight analysis (`--skip-activation` for fast runs)

For each `(layer L, module ∈ {q_proj, v_proj})`:

1. Compute `ΔW = B @ A` (the effective weight update)
2. SVD: `U, S, Vt = svd(ΔW)`
3. `cos_sim_input`  — cosine similarity between `Vt[0,:]` and the steering `first_direction` in input space
4. `cos_sim_output` — cosine similarity between `U[:,0]` and the steering `first_direction` in output space (q\_proj only; v\_proj output dim ≠ hidden\_size)

### Activation analysis

Runs forward passes on a set of prompts through both the base model and the LoRA model. For each layer:

- `Δh = mean(h_lora) − mean(h_base)` averaged over prompts and tokens
- `cos_sim_first_direction` — cosine similarity of `Δh` with the steering `first_direction`

### Directions files

The analysis reads `.npy` files produced by `pytorch_pure/extract_directions.py`. The auto-discovery order is:

1. `pytorch_pure/output/{model_short}/steering_config-en-{dir_id}-pca_0.npy`
2. `output/{model_short}/steering_config-en-dir_{dir_id}-pca_0.npy` (may be a git-LFS pointer)

Pass `--directions-file` to override. If a git-LFS pointer is detected the tool prints the pull command and exits cleanly.

---

## Dependencies

```
torch >= 2.0
transformers >= 4.40
peft >= 0.10       # LoRA
safetensors        # weight loading in analyze/
numpy
pyyaml
tqdm
vllm               # prepare stage and eval --backend vllm
```

Install:

```bash
pip install -r lora/requirements.txt
```
