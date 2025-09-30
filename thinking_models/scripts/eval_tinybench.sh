#!/bin/bash

# Parse command line arguments
PORT=""
IDX=""
TOTAL=""
ANGLE_INTERVAL=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --idx)
            IDX="$2"
            shift 2
            ;;
        --total)
            TOTAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Function to get angles for this job
get_angles_for_job() {
    local idx=$1
    local total=$2

    if [ "$total" == "1" ]; then
        echo "none $(seq 0 $ANGLE_INTERVAL 350)"
    elif [ "$total" == "4" ]; then
        # Based on the commented ranges in original file
        case $idx in
            0) echo "none $(seq 0 $ANGLE_INTERVAL 60)" ;;
            1) echo "$(seq 90 $ANGLE_INTERVAL 150)" ;;
            2) echo "$(seq 180 $ANGLE_INTERVAL 210)" ;;
            3) echo "$(seq 240 $ANGLE_INTERVAL 330)" ;;
        esac
    elif [ "$total" == "6" ]; then
        # Based on the commented ranges in original file
        case $idx in
            0) echo "none $(seq 0 $ANGLE_INTERVAL 0)" ;;
            1) echo "$(seq 30 $ANGLE_INTERVAL 90)" ;;
            2) echo "$(seq 120 $ANGLE_INTERVAL 150)" ;;
            3) echo "$(seq 180 $ANGLE_INTERVAL 180)" ;;
            4) echo "$(seq 210 $ANGLE_INTERVAL 240)" ;;
            5) echo "$(seq 270 $ANGLE_INTERVAL 330)" ;;
        esac
    elif [ "$total" == "8" ]; then
        # Based on the commented ranges in original file
        case $idx in
            0) echo "none $(seq 0 $ANGLE_INTERVAL 0)" ;;
            1) echo "$(seq 30 $ANGLE_INTERVAL 60)" ;;
            2) echo "$(seq 90 $ANGLE_INTERVAL 120)" ;;
            3) echo "$(seq 150 $ANGLE_INTERVAL 150)" ;;
            4) echo "$(seq 180 $ANGLE_INTERVAL 180)" ;;
            5) echo "$(seq 210 $ANGLE_INTERVAL 210)" ;;
            6) echo "$(seq 240 $ANGLE_INTERVAL 270)" ;;
            7) echo "$(seq 300 $ANGLE_INTERVAL 330)" ;;
        esac
    else
        echo "Unsupported total: $total. Supported values: 1, 4, 6, 8"
        exit 1
    fi
}

TASKS=(
    # tinyBenchmarks
    # tinyHellaswag
    aime25
    tinyArc
    tinyGSM8k
    # tinyMMLU
    # tinyTruthfulQA
    # tinyWinogrande
    gpqa_diamond_cot_zeroshot
    humaneval_instruct
)

# needed for humaneval_instruct
export HF_ALLOW_CODE_EVAL="1"

declare -A EXTRACTION_POINT_ID

# Populate the nested structure
EXTRACTION_POINT_ID["Qwen/Qwen3-4B::s1_start_n_end"]="post_attention_layernorm_14"
EXTRACTION_POINT_ID["Qwen/Qwen3-4B::s2_start_2_n_end_2"]="post_attention_layernorm_23"
EXTRACTION_POINT_ID["Qwen/Qwen3-4B::s3_mean_thinking_n_mean_answer"]="input_layernorm_22"
EXTRACTION_POINT_ID["Qwen/Qwen3-4B::s4_vanilla_n_fast_thinking"]="input_layernorm_25"

EXTRACTION_POINT_ID["Qwen/Qwen3-8B::s1_start_n_end"]="post_attention_layernorm_16"
EXTRACTION_POINT_ID["Qwen/Qwen3-8B::s2_start_2_n_end_2"]="input_layernorm_25"
EXTRACTION_POINT_ID["Qwen/Qwen3-8B::s3_mean_thinking_n_mean_answer"]="input_layernorm_22"
EXTRACTION_POINT_ID["Qwen/Qwen3-8B::s4_vanilla_n_fast_thinking"]="input_layernorm_23"

EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B::s1_start_n_end"]="input_layernorm_15"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B::s2_start_2_n_end_2"]="input_layernorm_19"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B::s3_mean_thinking_n_mean_answer"]="input_layernorm_12"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B::s4_vanilla_n_fast_thinking"]="post_attention_layernorm_16"

EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-LLama-8B::s1_start_n_end"]="post_attention_layernorm_18"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-LLama-8B::s2_start_2_n_end_2"]="post_attention_layernorm_22"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-LLama-8B::s3_mean_thinking_n_mean_answer"]="post_attention_layernorm_20"
EXTRACTION_POINT_ID["deepseek-ai/DeepSeek-R1-Distill-LLama-8B::s4_vanilla_n_fast_thinking"]="post_attention_layernorm_19"

# Function to get extraction point
get_extraction_point() {
    local model="$1"
    local strategy="$2"
    echo "${EXTRACTION_POINT_ID["$model::$strategy"]}"
}


MODELS=(
    # "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "deepseek-ai/DeepSeek-R1-Distill-LLama-8B"
)

# Define models with their corresponding ports
declare -A MODEL_PORTS=(
    ["Qwen/Qwen3-4B"]=9907
    # ["Qwen/Qwen3-8B"]=9904
    # ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]=9901
    # ["deepseek-ai/DeepSeek-R1-Distill-LLama-8B"]=9907
)



EXTRACTION_STRATEGIES=(
    "s4_vanilla_n_fast_thinking"
    "s3_mean_thinking_n_mean_answer"
    "s1_start_n_end"
    "s2_start_2_n_end_2"
)

for model_id in "${MODELS[@]}"; do
    for extraction_strategy_id in "${EXTRACTION_STRATEGIES[@]}"; do
        extraction_point_id=$(get_extraction_point "$model_id" "$extraction_strategy_id")
        echo "Using extraction point: $extraction_point_id for model: $model_id"

        # Use provided port or fall back to MODEL_PORTS
        if [ -n "$PORT" ]; then
            port=$PORT
        else
            port=${MODEL_PORTS[$model_id]}
        fi

        for task in "${TASKS[@]}"; do
            # baseline
            model_name=$(echo "$model_id" | cut -d'/' -f2)

            dir_id="${extraction_strategy_id}"

            # Get angles for this job
            if [ -n "$IDX" ] && [ -n "$TOTAL" ]; then
                angles=($(get_angles_for_job $IDX $TOTAL))
            else
                angles=(none $(seq 0 $ANGLE_INTERVAL 350))
            fi

            # steer
            for angle in "${angles[@]}"; do
                output_dir=./outputs/thinking-steering-benchmarks/${model_name}/${extraction_point_id}/${task}/adaptive_${angle}
                endpoint_url=http://0.0.0.0:${port}/thinking_steering/${extraction_point_id}/${angle}
                wandb_id=${model_name}+${task}+adaptive_${angle}+${extraction_point_id}

                # Check if results already exist
                if [ -d "$output_dir" ]; then
                    echo "Results already exist for ${output_dir}, skipping..."
                    continue
                fi

                echo "Evaluating model: $model_id with angle: $angle on port $port"
                time lm_eval \
                    --model local-completions \
                    --tasks ${task} \
                    --batch_size 1 \
                    --model_args model=${model_id},base_url=${endpoint_url},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
                    --output_path ${output_dir} \
                    --wandb_args project=lm-eval-thinking-steering,entity=lone17,id=${wandb_id} \
                    --log_samples \
                    --confirm_run_unsafe_code
            done
        done
    done
done