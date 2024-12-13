export OPENAI_API_KEY="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"

# Define the tuples
declare -a tuples=(
    "2 4"
    "3 5"
    "4 6"
    "5 7"
)

# Loop over the tuples
for tuple in "${tuples[@]}"; do
    read -r code_length num_colors <<< "$tuple"
    
    python run.py \
        --model gpt-4o \
        --model_type openai \
        --code_length "$code_length" \
        --num_colors "$num_colors" \
        --num_runs 100 \
        --save_results
done
