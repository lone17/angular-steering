#!/bin/bash

STEERING_VECTORS_PATH=${STEERING_VECTORS_PATH:-outputs/steering-vectors/Qwen3-4B/candidate_steering_dirs.pt}
UPPERBOUND_MAX_NEW_TOKENS=${UPPERBOUND_MAX_NEW_TOKENS:-4000}
DEVICE=${DEVICE:-cuda:0}

declare -A EXTRACTION_STRATEGIES=(
    # ["s1_start_n_end"]=52
    # ["s2_start_2_n_end_2"]=58
    ["s4_vanilla_n_fast_thinking"]=43
)

for strategy_id in "${!EXTRACTION_STRATEGIES[@]}"; do
    chosen_extraction_point=${EXTRACTION_STRATEGIES[$strategy_id]}
    echo "Applying angular steering with strategy: $strategy_id"
    time python iv_angular_steering.py \
        --model_id Qwen/Qwen3-4B \
        --device $DEVICE \
        --upperbound_max_new_tokens $UPPERBOUND_MAX_NEW_TOKENS \
        --random_seed 42 \
        --root_output_dir outputs/ \
        --module_names input_layernorm post_attention_layernorm \
        --chosen_extraction_point $chosen_extraction_point \
        --extraction_strategy_id $strategy_id \
        --target_angle $(seq 165 30 170) \
        --adaptive_mode 1 \
        --force_overwrite_output \
        --end_index 3
done