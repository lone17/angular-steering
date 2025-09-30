#!/bin/bash

MODELS=(
    "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "deepseek-ai/DeepSeek-R1-Distill-LLama-8B"
)

for model_id in "${MODELS[@]}"; do
    model_name=${model_id##*/}

    python get_prompt_activations.py \
        --model_id ${model_id} \
        --device cuda \
        --batch_size 512 \
        --output_dir outputs/thinking-model-activations/${model_name} \
        --module_names input_layernorm post_attention_layernorm \
        --offload_to_cpu \
        --start_index 0 \
        --think_fast
done
