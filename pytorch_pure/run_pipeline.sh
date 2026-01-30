#!/bin/bash
# Pure PyTorch Angular Steering Pipeline with configurable parameters

set -e

# Default configuration (matches parent implementation)
# MODEL="google/gemma-2-9b-it"
MODEL="qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR="./output"
LANGUAGE="en"
N_SAMPLES=512              # 512 for proper layer selection (128 too small)
BATCH_SIZE=4
MAX_TOKENS=256
ANGLE_STEP=10              # 10° steps for full evaluation (36 angles)
ADAPTIVE_MODE=1            # Conditional steering (matches parent)
STRATEGY="max_sim"         # Max similarity strategy
PROMPT_ONLY=false          # Steering mode: false=all tokens, true=prompt only

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --angle-step)
            ANGLE_STEP="$2"
            shift 2
            ;;
        --adaptive-mode)
            ADAPTIVE_MODE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --prompt-only)
            PROMPT_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL            Model ID (default: google/gemma-2-9b-it)"
            echo "  --output-dir DIR         Output directory (default: ./output)"
            echo "  --language LANG          Language: en (default: en)"
            echo "  --n-samples N            Number of samples for direction extraction (default: 512)"
            echo "                           IMPORTANT: Use 512+ for proper layer selection, not 128!"
            echo "  --batch-size N           Batch size (default: 4)"
            echo "  --max-tokens N           Max tokens per generation (default: 256)"
            echo "  --angle-step N           Angle step in degrees (default: 10)"
            echo "                           10 for full eval (36 angles), 30 for quick test (12 angles)"
            echo "  --adaptive-mode N        Steering mode: 0=always, 1=conditional (default: 1)"
            echo "  --strategy STRATEGY      Direction strategy: max_sim, max_norm, both (default: max_sim)"
            echo "  --prompt-only            Apply steering only to prompt (matches vLLM enforce_eager=False)"
            echo ""
            echo "Examples:"
            echo "  # Default run"
            echo "  $0"
            echo ""
            echo "  # Quick test with 30-degree steps"
            echo "  $0 --angle-step 30 --n-samples 256"
            echo ""
            echo "  # Different model"
            echo "  $0 --model Qwen/Qwen2.5-7B-Instruct"
            echo ""
            echo "  # Prompt-only steering (steers prompt, cached K/V influence generation)"
            echo "  $0 --prompt-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "==============================================="
echo "PyTorch Pure Angular Steering Pipeline"
echo "==============================================="
echo "  Model:         $MODEL"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  Language:      $LANGUAGE"
echo "  Samples:       $N_SAMPLES (for direction extraction)"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Max Tokens:    $MAX_TOKENS"
echo "  Angle Step:    $ANGLE_STEP degrees"
echo "  Adaptive Mode: $ADAPTIVE_MODE"
echo "  Strategy:      $STRATEGY"
echo "  Prompt Only:   $PROMPT_ONLY"
echo ""

# Step 1: Extract Directions
echo "Step 1/4: Extracting steering directions..."
uv run python extract_directions.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --language "$LANGUAGE" \
    --n-samples "$N_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --strategy "$STRATEGY"

echo ""
echo "✓ Direction extraction complete!"
echo ""

# Step 2: Generate Responses
echo "Step 2/4: Generating responses with steering..."

# Determine strategy filter for response generation
if [ "$STRATEGY" = "both" ]; then
    STRATEGY_FILTER="max_sim,max_norm"
else
    STRATEGY_FILTER="$STRATEGY"
fi

# Build prompt-only flag
PROMPT_ONLY_FLAG=""
if [ "$PROMPT_ONLY" = true ]; then
    PROMPT_ONLY_FLAG="--prompt-only"
fi

uv run python generate_responses.py \
    --model "$MODEL" \
    --config-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --adaptive-mode "$ADAPTIVE_MODE" \
    --language "$LANGUAGE" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --angle-step "$ANGLE_STEP" \
    --strategy-filter "$STRATEGY_FILTER" \
    $PROMPT_ONLY_FLAG

echo ""
echo "✓ Response generation complete!"
echo ""

echo "========================================="
echo "PyTorch Pure Pipeline Complete!"
echo "========================================="
echo ""
echo "✓ Steering directions extracted"
echo "✓ Responses generated with angular steering"
echo ""
echo "Results saved to: $OUTPUT_DIR/$(basename $MODEL)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next Steps - Evaluation & Visualization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Copy outputs to parent directory:"
echo "   cp -r $OUTPUT_DIR/* ../output/"
echo ""
echo "2. Run evaluation scripts (from parent directory):"
echo "   cd .."
echo "   python evaluate_jailbreak.py"
echo "   python eval_perplexity.py"
echo ""
echo "3. Visualize results:"
echo "   jupyter notebook visualization.ipynb"
echo ""
echo "See parent README.md for full pipeline details."
echo ""
