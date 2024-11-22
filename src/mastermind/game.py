import random
from typing import List, Tuple

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
Hint = str


class Mastermind:
    def __init__(
        self,
        code_length: int = 4,
        num_colors: int = 6,
        max_guesses: int = 12,
        duplicates_allowed: bool = True,
    ):
        self.code_length = code_length
        self.max_guesses = max_guesses
        self.num_colors = num_colors
        self.duplicates_allowed = duplicates_allowed
        self.possible_colors: List[str] = random.sample(COLORS, k=num_colors)
        self.secret_code = self._generate_secret_code()

    def _generate_secret_code(self) -> List[str]:
        if self.duplicates_allowed:
            return random.choices(self.possible_colors, k=self.code_length)
        else:
            return random.sample(self.possible_colors, k=self.code_length)

    def compute_all_possible_scores(self) -> List[Tuple[int, int]]:
        all_possible_scores = [
            (black, white)
            for black in range(self.code_length + 1)  # `black` can range from 0 to `code_length`
            for white in range(self.code_length - black + 1)  # `white` depends on remaining slots
            if black + white <= self.code_length  # Total feedback must not exceed `code_length`
        ]
        return all_possible_scores

    def evaluate_guess(self, guess: List[str], code: List[str]) -> Tuple[int, int]:
        exact_matches = sum(s == g for s, g in zip(code, guess))
        partial_matches = sum(min(code.count(color), guess.count(color)) for color in set(code)) - exact_matches
        return exact_matches, partial_matches

    def evaluate(self, guess: List[str]) -> Tuple[ExactMatches, PartialMatches, Hint]:
        exact_matches, partial_matches = self.evaluate_guess(guess, self.secret_code)
        return (
            exact_matches,
            partial_matches,
            f"Correct color and position: {exact_matches}. Correct color but wrong position: {partial_matches}.",
        )

    def reset(self):
        self.possible_colors = random.sample(COLORS, k=self.num_colors)
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
