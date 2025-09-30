#!/bin/bash

UPPERBOUND_MAX_NEW_TOKENS=${UPPERBOUND_MAX_NEW_TOKENS:-16000}

MODELS=(
    # Qwen/Qwen3-4B
    Qwen/Qwen3-8B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-LLama-8B
)

for model_id in "${MODELS[@]}"; do
    echo "Getting activations for model: $model_id"
    model_name=$(echo "$model_id" | cut -d'/' -f2)
    time python step1_get_activations.py \
        --model_id $model_id \
        --device cuda:0 \
        --upperbound_max_new_tokens $UPPERBOUND_MAX_NEW_TOKENS \
        --random_seed 42 \
        --output_dir outputs/thinking-model-activations/$model_name \
        --module_names input_layernorm post_attention_layernorm \
        --start_index 125 \
        --add_think_fast_prompt \
        --end_index 250
        # --disable_thinking \
done