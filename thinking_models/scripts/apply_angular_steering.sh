#!/bin/bash

CHOSEN_EXTRACTION_POINT=${CHOSEN_EXTRACTION_POINT:-52}
STEERING_VECTORS_PATH=${STEERING_VECTORS_PATH:-output/Qwen3-4B/candidate_steering_dirs.pt}
UPPERBOUND_MAX_NEW_TOKENS=${UPPERBOUND_MAX_NEW_TOKENS:-32000}
DEVICE=${DEVICE:-cuda:0}

python iv_angular_steering.py \
    --model_id Qwen/Qwen3-4B \
    --device $DEVICE \
    --upperbound_max_new_tokens $UPPERBOUND_MAX_NEW_TOKENS \
    --random_seed 42 \
    --output_dir outputs/Qwen3-4B \
    --module_names input_layernorm post_attention_layernorm \
    --steering_vectors_path $STEERING_VECTORS_PATH \
    --chosen_extraction_point $CHOSEN_EXTRACTION_POINT
