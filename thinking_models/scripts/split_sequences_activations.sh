#!/bin/bash

MODELS=(
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-LLama-8B"
)

GENERATION_STRATEGIES=(
    "activations"
    "activations_no_thinking"
    "activations_with_think_fast_prompt"
)

for model_id in "${MODELS[@]}"; do
    model_name=${model_id##*/}

    for generation_strategy in "${GENERATION_STRATEGIES[@]}"; do
        if [ "$generation_strategy" == "activations" ]; then
            output_name="generation_default"
        else
            if [ "$generation_strategy" == "activations_no_thinking" ]; then
                output_name="generation_no_thinking"
            else
                if [ "$generation_strategy" == "activations_with_think_fast_prompt" ]; then
                    output_name="generation_with_think_fast_prompt"
                fi
            fi
        fi

        python split_activations.py \
            --acts_dir outputs/thinking-model-activations/${model_name}/${generation_strategy}/ \
            --output_dir outputs/thinking-model-activations/${model_name}/${output_name}/ \
            --remove_original

    done
done
