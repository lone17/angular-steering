#!/bin/bash

MODEL_ID="Qwen/Qwen3-4B"
# MODEL_ID="Qwen/Qwen3-8B"
# MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-LLama-8B"

MODEL_NAME=${MODEL_ID##*/}


python split_activations.py \
    --acts_dir outputs/${MODEL_NAME}/ \
    --output_dir outputs/${MODEL_NAME}/ \
    --remove_original \

