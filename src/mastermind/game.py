import random
from collections import Counter
from itertools import product
from typing import Dict, List, Tuple

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]

ExactMatches = int
PartialMatches = int
Progress = float
Hint = str


class Mastermind:
    def __init__(
        self, code_length: int = 4, num_colors: int = 6, max_guesses: int = 12, duplicates_allowed: bool = True
    ):
        self.code_length = code_length
        self.max_guesses = max_guesses
        self.num_colors = num_colors
        self.possible_colors = random.sample(COLORS, k=num_colors)
        self.duplicates_allowed = duplicates_allowed
        self.secret_code = self._generate_secret_code()
        self.progress_lookup = self.precompute_progress()

    def _generate_secret_code(self) -> List[str]:
        if self.duplicates_allowed:
            return random.choices(self.possible_colors, k=self.code_length)
        else:
            return random.sample(self.possible_colors, k=self.code_length)

    def precompute_progress(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        all_possible_codes = list(product(self.possible_colors, repeat=self.code_length))
        all_hints = Counter([self.evaluate_guess(code) for code in all_possible_codes])

        progress_lookup = {}
        for key in all_hints.keys():
            cumulative_progress = sum(value for k, value in all_hints.items() if k[0] >= key[0] and k[1] >= key[1])
            progress_lookup[key] = {
                "percent": 1 - (cumulative_progress / len(all_possible_codes)) if not cumulative_progress == 1 else 1.0,
                "remaining_states": cumulative_progress,
            }

        return progress_lookup

    def evaluate_guess(self, guess: List[str]) -> Tuple[int, int]:
        exact_matches = sum(s == g for s, g in zip(self.secret_code, guess))
        partial_matches = (
            sum(min(self.secret_code.count(color), guess.count(color)) for color in set(self.secret_code))
            - exact_matches
        )

        return exact_matches, partial_matches

    def evaluate(self, guess: List[str]) -> Tuple[ExactMatches, PartialMatches, Progress, Hint]:
        exact_matches, partial_matches = self.evaluate_guess(guess)
        progress = self.progress_lookup.get((exact_matches, partial_matches), 0.0)

        return (
            exact_matches,
            partial_matches,
            progress,
            f"Correct color and position: {exact_matches}. Correct color but wrong position: {partial_matches}.",
        )

    def reset(self):
        self.secret_code = self._generate_secret_code()

    def to_json(self):
        return {
            "code_length": self.code_length,
            "possible_colors": self.possible_colors,
            "duplicates_allowed": self.duplicates_allowed,
            "secret_code": self.secret_code,
        }

    def __repr__(self):
        return (
            f"<Mastermind(possible_colors={len(self.possible_colors)}, code_length={self.code_length}, "
            f"color_names={self.possible_colors}, secret_code_hidden=True)>"
        )
