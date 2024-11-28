CUDA_VISIBLE_DEVICES=0 \
    python run.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --model_type hf \
    --code_length 4 \
    --num_colors 6 \
    --num_runs 100 \
    --save_results True
