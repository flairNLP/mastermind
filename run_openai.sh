export OPENAI_API_KEY="[API key here]"

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
