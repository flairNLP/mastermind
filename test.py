import random
from itertools import product

# Define all possible codes (4 pegs, 6 colors)
colors = ['red', 'green', 'blue', 'yellow', 'grey', 'pink']
all_codes = list(product(colors, repeat=4))


# Function to compute feedback
def compute_feedback(guess, code):
    black = sum(g == c for g, c in zip(guess, code))
    white = sum(min(guess.count(color), code.count(color)) for color in set(colors)) - black
    return black, white


# Random guess
guesses = [
    (['red', 'blue', 'pink', 'green'], (1, 2)),
    (['blue', 'green', 'yellow', 'pink'], (1, 2)),
    (['grey', 'red', 'yellow', 'blue'], (0, 2)),
    (['red', 'pink', 'blue', 'green'], (1, 2)),
]

# Filter codes
remaining_codes = []
for code in all_codes:
    if all(compute_feedback(guess, code) == feedback for guess, feedback in guesses):
        remaining_codes.append(code)

print(f"Remaining possible codes: {remaining_codes}")
