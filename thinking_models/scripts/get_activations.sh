#!/bin/bash

UPPERBOUND_MAX_NEW_TOKENS=32000

python i_get_activations.py \
    --model_id Qwen/Qwen3-4B \
    --device cuda:0 \
    --batch_size 512 \
    --upperbound_max_new_tokens $UPPERBOUND_MAX_NEW_TOKENS \
    --random_seed 42 \
    --output_dir outputs/Qwen3-4B \
    --module_names input_layernorm post_attention_layernorm
