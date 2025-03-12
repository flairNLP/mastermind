#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# export ANTHROPIC_API_KEY=[key here]
# export OPENAI_API_KEY=[key here]

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <flag1> <flag2> ..."
    echo "Example: $0 hf"
    exit 1
fi

for flag in "$@"; do
    case $flag in
        hf)
            echo "Executing tasks for Hugging Face..."
            hf_models=(
                "Qwen/Qwen2.5-3B-Instruct"
                "Qwen/Qwen2.5-7B-Instruct"
                "meta-llama/Llama-3.2-3B-Instruct"
                "meta-llama/Llama-3.1-8B-Instruct"
                "microsoft/Phi-3.5-mini-instruct"
                "microsoft/phi-4"
                'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            )

            for model in "${hf_models[@]}"; do
                model_path="${model##*/}"
                for dataset in "${datasets[@]}"; do
                    echo "Running script for HF model: $model"
                    lm-eval --model hf --model_args "pretrained=$model" --tasks "mastermind" --output_path "results/eval_harness"
                done
            done
            ;;
        *)
            echo "Unknown flag: $flag"
            ;;
    esac
done
