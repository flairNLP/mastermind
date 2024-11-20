CUDA_VISIBLE_DEVICES=0 \
    python run.py \
    --model gpt-4-turbo-preview \
    --model_type openai \
    --generation_kwargs '{"max_new_tokens": 1024, "temperature": 0.7}' \
    --code_length 4 \
    --num_colors 6 \
    --duplicates_allowed \
    --num_runs 100 \
    --save_results True
