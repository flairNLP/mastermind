#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# export ANTHROPIC_API_KEY=[key here]
# export OPENAI_API_KEY=[key here]

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <flag1> <flag2> ..."
    echo "Example: $0 hf openai anthropic"
    exit 1
fi

# Dataset path
# Loop through all provided flags
datasets=(
    "flair/mastermind_24_prompt"
    "flair/mastermind_36_prompt"
    "flair/mastermind_46_prompt"
)

for flag in "$@"; do
    case $flag in
        hf)
            echo "Executing tasks for Hugging Face..."
            hf_models=(
                "Qwen/Qwen3-4B"
            )

            for model in "${hf_models[@]}"; do
                for dataset in "${datasets[@]}"; do
                    echo "Running script for HF model: $model"
                    python run_instructions.py --model_type "hf" --model "$model" --dataset "$dataset" --enable_thinking false
                done
            done
            ;;
        openai)
            echo "Executing tasks for OpenAI..."
            openai_models=(
            #    "gpt-4o"
            #    "gpt-4o-mini"
                # "o1"
                "o3-mini"
            )

            for model in "${openai_models[@]}"; do
                for dataset in "${datasets[@]}"; do
                    echo "Running script for OpenAI model: $model"
                    python run_instructions.py --model_type "openai" --model "$model" --dataset "$dataset"
                done
            done
            ;;
        anthropic)
            echo "Executing tasks for Anthropic..."
            anthropic_models=(
                "claude-3-5-sonnet-latest"
            )

            for model in "${anthropic_models[@]}"; do
                for dataset in "${datasets[@]}"; do
                    echo "Running script for Anthropic model: $model"
                    python run_instructions.py --model_type "anthropic" --model "$model" --dataset "$dataset"
                done
            done
            ;;
        *)
            echo "Unknown flag: $flag"
            ;;
    esac
done
