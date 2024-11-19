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

EXACT_MATCHES = int
PARTIAL_MATCHES = int
HINT = str


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

    def _generate_secret_code(self) -> List[str]:
        if self.duplicates_allowed:
            return random.choices(self.possible_colors, k=self.code_length)
        else:
            return random.sample(self.possible_colors, k=self.code_length)

    def evaluate_guess(self, guess: List[str]) -> Tuple[int, int]:
        exact_matches = sum(s == g for s, g in zip(self.secret_code, guess))
        partial_matches = (
            sum(min(self.secret_code.count(color), guess.count(color)) for color in set(self.secret_code))
            - exact_matches
        )

        return exact_matches, partial_matches

    def evaluate(self, guess: List[str]) -> Tuple[EXACT_MATCHES, PARTIAL_MATCHES, HINT]:
        exact_matches, partial_matches = self.evaluate_guess(guess)

        return (
            exact_matches,
            partial_matches,
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
