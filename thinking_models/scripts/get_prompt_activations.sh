#!/bin/bash

# MODEL_ID="Qwen/Qwen3-4B"
# MODEL_ID="Qwen/Qwen3-8B"
MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-LLama-8B"

MODEL_NAME=${MODEL_ID##*/}

python get_prompt_activations.py \
    --model_id ${MODEL_ID} \
    --device cuda \
    --batch_size 512 \
    --output_dir outputs/${MODEL_NAME} \
    --module_names input_layernorm post_attention_layernorm \
    --offload_to_cpu \
    --start_index 0 \
    # --think_fast \
