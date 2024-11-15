import random
from enum import Enum, auto
from typing import List, Tuple, Union

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


class GameState(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Mastermind:
    def __init__(
        self, code_length: int = 4, num_colors: int = 6, max_guesses: int = 12, duplicates_allowed: bool = True
    ):
        """
        Initializes the Mastermind game.

        Args:
            num_colors (int): Number of available colors to choose from.
            code_length (int): Length of the secret code to guess.
            color_names (Optional[List[str]]): Names of the colors. If None, default names are generated.
        """
        self.code_length = code_length
        self.max_guesses = max_guesses
        self.num_colors = num_colors
        self.possible_colors = random.sample(COLORS, k=num_colors)
        self.duplicates_allowed = duplicates_allowed
        self.attempts = 0
        self.task_instruction = (
            f"You task is to solve the game of Mastermind. The game is defined as follows:\n"
            f"- You have to find out the {self.code_length}-color secret code.\n"
            f"- The following colors are allowed: {self.possible_colors}.\n"
            f"- Duplicates colors are{' ' if self.duplicates_allowed else ' not '}allowed.\n"
            f"- You have in total {self.max_guesses} tries.\n"
            "- After each guess, I will let you know whether how many of your guesses are the correct color and in the correct position and how many are the correct colors but in the wrong position.\n\n"
            f"Your input needs to be a list of strings, e.g.:"
            f"Input: {random.sample(self.possible_colors, k=self.code_length)}"
            f"Hint: Correct color and position: 1. Correct color but wrong position: 1."
        )
        self.secret_code = self._generate_secret_code()
        self.status = GameState.ONGOING

    def _generate_secret_code(self) -> List[str]:
        """
        Generates a random secret code.

        Returns:
            List[str]: The generated secret code.
        """
        if self.duplicates_allowed:
            return random.choices(self.possible_colors, k=self.code_length)
        else:
            return random.sample(self.possible_colors, k=self.code_length)

    def evaluate_guess(self, guess: List[str]) -> Tuple[int, int]:
        """
        Evaluates the player's guess.

        Args:
            guess (List[str]): The player's guessed code.

        Returns:
            Tuple[int, int]: A tuple containing:
                - Number of correct colors in the correct position (exact matches).
                - Number of correct colors in the wrong position (partial matches).
        """
        exact_matches = sum(s == g for s, g in zip(self.secret_code, guess))
        partial_matches = (
            sum(min(self.secret_code.count(color), guess.count(color)) for color in set(self.secret_code))
            - exact_matches
        )
        return exact_matches, partial_matches

    def make_guess(self, guess: List[str]) -> Union[str, None]:
        """
        Makes a guess and evaluates it.

        Args:
            guess (List[str]): The player's guessed code.

        Returns:
            str: Result of the evaluation (exact matches, partial matches).
        """
        if self.state != GameState.ONGOING:
            raise ValueError("Game is already over.")

        self.attempts += 1

        if len(guess) != self.code_length:
            raise ValueError(f"Guess must be of length {self.code_length}.")

        exact_matches, partial_matches = self.evaluate_guess(guess)

        if exact_matches == self.code_length:
            self.state = GameState.WON
        elif self.attempts >= self.max_guesses:
            self.state = GameState.LOST
        else:
            return f"Correct color and position: {exact_matches}. Correct color but wrong position: {partial_matches}."

    def __repr__(self):
        return (
            f"<Mastermind(num_colors={self.num_colors}, code_length={self.code_length}, "
            f"color_names={self.possible_colors}, secret_code_hidden=True)>"
        )
