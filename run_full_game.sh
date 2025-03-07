#!/bin/bash

# Set GPU device
# export CUDA_VISIBLE_DEVICES=0
# export ANTHROPIC_API_KEY=[key here]
# export OPENAI_API_KEY=[key here]

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <flag1> <flag2> ..."
    echo "Example: $0 hf openai anthropic"
    exit 1
fi

# Define arrays for iteration
declare -a tuples=(
    "2 4"
    "3 5"
    "4 6"
    "5 7"
    "6 8"
    "7 9"
)


# Iterate over models, code lengths, and number of colors
for flag in "$@"; do
    case $flag in
        hf)
            echo "Executing tasks for Hugging Face..."
            hf_models=(
                "Qwen/Qwen2-1.5B-Instruct"
                "Qwen/Qwen2.5-3B-Instruct"
                "Qwen/Qwen2.5-7B-Instruct"
                "meta-llama/Llama-3.2-3B-Instruct"
                "meta-llama/Llama-3.1-8B-Instruct"
                "microsoft/Phi-3.5-mini-instruct"
                "microsoft/phi-4"
                'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            )

            for i in "${!hf_models[@]}"; do
                model="${hf_models[$i]}"
            
                # Loop over the tuples
                for tuple in "${tuples[@]}"; do
                    read -r code_length num_colors <<< "$tuple"
                    echo "Running model: $model, code_length: $code_length, num_colors: $num_colors"
                
                    python run_full_game.py \
                        --model "$model" \
                        --model_type "hf" \
                        --code_length "$code_length" \
                        --num_colors "$num_colors" \
                        --num_runs 100 \
                        --save_results
                done
            done
            ;;
        openai)
            echo "Executing tasks for OpenAI..."
            openai_models=(
                "gpt-4o"
                "gpt-4o-mini"
                "o3-mini"
            )

            for i in "${!openai_models[@]}"; do
                model="${openai_models[$i]}"
                for tuple in "${tuples[@]}"; do
                    read -r code_length num_colors <<< "$tuple"
                    echo "Running script for OpenAI model: $model"
                    
                    python run_full_game.py \
                        --model_type "openai" \
                        --model "$model" \
                        --code_length "$code_length" \
                        --num_colors "$num_colors" \
                        --num_runs 100 \
                        --save_results
                done
            done
            ;;
        anthropic)
            echo "Executing tasks for Anthropic..."
            anthrophic_models=(
                "haiku-3-5-haiku-latest"
                "claude-3-5-sonnet-latest"
            )

            for i in "${!anthropic_models[@]}"; do
                model="${anthropic_models[$i]}"
                for tuple in "${tuples[@]}"; do
                    read -r code_length num_colors <<< "$tuple"
                    echo "Running script for Anthropic model: $model"
                    
                    python run_full_game.py \
                        --model_type "anthropic" \
                        --model "$model" \
                        --code_length "$code_length" \
                        --num_colors "$num_colors" \
                        --num_runs 100 \
                        --save_results
                done
            done
            ;;
        *)
            echo "Unknown flag: $flag"
            ;;
    esac
done
