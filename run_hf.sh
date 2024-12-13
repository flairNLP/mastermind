#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=2

# Define arrays for iteration
#models=("Qwen/Qwen2-1.5B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "microsoft/Phi-3.5-mini-instruct")
models=(
    "microsoft/Phi-3-small-8k-instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "allenai/Llama-3.1-Tulu-3-8B"
)
declare -a tuples=(
    "2 4"
    "3 5"
    "4 6"
    "5 7"
)

# Iterate over models, code lengths, and number of colors
for i in "${!models[@]}"; do
    model="${models[$i]}"
  
    # Loop over the tuples
    for tuple in "${tuples[@]}"; do
        read -r code_length num_colors <<< "$tuple"
        echo "Running model: $model, code_length: $code_length, num_colors: $num_colors"
      
        python run.py \
            --model "$model" \
            --model_type "hf" \
            --code_length "$code_length" \
            --num_colors "$num_colors" \
            --num_runs 100 \
            --save_results
    done
done

echo "All runs completed!"