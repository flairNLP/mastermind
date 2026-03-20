#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# export ANTHROPIC_API_KEY=[key here]
# export OPENAI_API_KEY=[key here]

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <flag1> <flag2> ..."
    echo "Example: $0 hf"
    exit 1
fi

sanitize_model_name() {
    echo "$1" | tr '/:' '__'
}

for flag in "$@"; do
    case $flag in
        hf)
            echo "Executing tasks for Hugging Face..."
            hf_models=(
                "Qwen/Qwen3-4B"
            )

            for model in "${hf_models[@]}"; do
                sanitized_model="$(sanitize_model_name "$model")"
                output_dir="results/multiple_choice/${sanitized_model}"
                echo "Running script for HF model: $model"
                mkdir -p "$output_dir"
                printf '{\n  "model": "%s"\n}\n' "$model" > "$output_dir/model_info.json"
                lm-eval --model hf --model_args "pretrained=$model" --tasks "mastermind" --output_path "$output_dir"
            done
            ;;
        *)
            echo "Unknown flag: $flag"
            ;;
    esac
done
