#!/bin/bash
export OPENAI_API_KEY="[API key here]"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

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
dataset="mastermind_46"

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
