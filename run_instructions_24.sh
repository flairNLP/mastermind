#!/bin/bash
export ANTHROPIC_API_KEY="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"
export OPENAI_API_KEY="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"
#export CUDA_VISIBLE_DEVICES=3

# List of models (HF Models)
hf_models=(
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/Phi-3-small-8k-instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "allenai/Llama-3.1-Tulu-3-8B"
)

# List of Anthropic Models
anthropic_models=(
    "claude-3-5-sonnet-latest"
)

# List of OpenAI Models
openai_models=(
    "gpt-4o"
)

# Dataset path
dataset="datasets/dataset_24_50k"

# Run the script for Hugging Face models
for model in "${hf_models[@]}"; do
    echo "Running script for HF model: $model"
    python instructions.py --model_type "hf" --model "$model" --dataset "$dataset"
done

# Run the script for Anthropic models
for model in "${anthropic_models[@]}"; do
    echo "Running script for Anthropic model: $model"
    python instructions.py --model_type "anthropic" --model "$model" --dataset "$dataset"
done

# Run the script for OpenAI models
for model in "${openai_models[@]}"; do
    echo "Running script for OpenAI model: $model"
    python instructions.py --model_type "openai" --model "$model" --dataset "$dataset"
done

