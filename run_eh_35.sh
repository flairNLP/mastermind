#!/bin/bash
export OPENAI_API_KEY="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"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=2

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

# List of OpenAI Models
openai_models=(
    "gpt-4o"
)

# Dataset path
dataset="mastermind_35"

# Run the script for Hugging Face models
#for model in "${hf_models[@]}"; do
#    echo "Running script for HF model: $model"
#    lm-eval --model hf --model_args "pretrained=$model" --tasks "$dataset" --output_path "results/$dataset/$model"
#done

# Run the script for OpenAI models
for model in "${openai_models[@]}"; do
    echo "Running script for OpenAI model: $model"
    lm-eval --model openai-completions --model_args "model=$model" --tasks "$dataset" --output_path "results/$dataset/$model"
done
