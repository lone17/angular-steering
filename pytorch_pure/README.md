# PyTorch Pure Implementation

**Pure PyTorch implementation of angular steering mechanism.**

This directory provides a PyTorch-only reference implementation without vLLM dependencies. For full pipeline documentation, algorithm explanation, and evaluation details, see the [parent README](../README.md).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run end-to-end pipeline
./run_pipeline.sh --model "Qwen/Qwen2.5-7B-Instruct"
```

## Pipeline Steps

### 1. Extract Directions

```bash
python extract_directions.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --n-samples 512 \
  --strategy max_sim
```

Extracts steering directions from model activations.

### 2. Generate Responses

```bash
# Full steering (all tokens)
python generate_responses.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --adaptive-mode 1 \
  --angle-step 10

# Prompt-only steering (matches vLLM enforce_eager=False)
python generate_responses.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --adaptive-mode 1 \
  --angle-step 10 \
  --prompt-only
```

Applies angular steering during generation at multiple angles.

**Steering modes:**
- Default: Steers all tokens during generation
- `--prompt-only`: Steers only prompt processing (first forward pass)
  - Steered K/V pairs are cached and reused during generation
  - Matches vLLM's `enforce_eager=False` behavior

### 3. Evaluate

```bash
# Copy outputs to parent directory
cp -r output/* ../output/

# Run evaluation scripts (from parent)
cd ..
python evaluate_jailbreak.py
python eval_perplexity.py

# Visualize results
jupyter notebook visualization.ipynb
```

## Files

- `utils.py` - Common utilities (data loading, tokenization, hooks)
- `extract_directions.py` - Direction extraction from activations
- `generate_responses.py` - Steered text generation
- `run_pipeline.sh` - Automated end-to-end pipeline
- `interactive_analysis.py` - Interactive notebook for comparing implementations

## Key Differences from Parent

- **Pure PyTorch** - No vLLM dependency
- **Modular** - Shared utilities in utils.py
- **Simplified** - Core steering implementation only
- **Educational** - Clear reference for understanding the mechanism

## Prompt-Only Steering

The `--prompt-only` flag enables steering that only affects prompt processing:

1. **First forward pass** (prompt): Steering is applied
2. **Generation passes**: No steering, but cached steered K/V pairs are reused

This matches vLLM's `enforce_eager=False` behavior where steering influences the model's "understanding" of the prompt without directly modifying generation.

For comprehensive documentation including what angular steering is, how it works, and full evaluation pipeline, refer to [the parent README](../README.md).
