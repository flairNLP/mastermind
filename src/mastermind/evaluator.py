import json
import random
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from mastermind.game import Mastermind
from mastermind.models import LanguageModel
from mastermind.utils import make_output_path, parse_guess

RESULT = Dict[str, str]


class GameState(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Evaluator:
    def __init__(self, game: Mastermind, model: LanguageModel):
        self.game = game
        self.model = model
        self.attempts = 0
        self.task_instruction = (
            f"Your task is to solve the game of Mastermind. The game is defined as follows:\n"
            f"- You have to find out the {game.code_length}-color secret code.\n"
            f"- The following colors are allowed: {game.possible_colors}.\n"
            f"- Duplicates colors are{' ' if game.duplicates_allowed else ' not '}allowed.\n"
            f"- You have in total {game.max_guesses} tries.\n"
            "- After each guess, I will let you know whether how many of your guesses are the correct color and in the correct position and how many are the correct colors but in the wrong position.\n\n"
            f"You must answer like this:\n"
            f"Guess: {random.sample(game.possible_colors, k=game.code_length)}\n\n"
            f"If your answer is not the secret code, I'll provide you with the following hint:\n"
            f"Hint: Correct color and position: 1. Correct color but wrong position: 1.\n###\n\n"
            "Guess: "
        )
        self.state = GameState.ONGOING

    def run(self, num_games: int = 1, save_results: bool = False, save_path: Optional[Path] = None) -> List[RESULT]:
        results = []
        for game in range(num_games + 1):
            chat_history = [{"role": "system", "content": self.task_instruction}]
            closest_score = 0
            closest_guess = None
            progress_bar = tqdm(total=self.game.max_guesses, desc="Attempts", unit="attempt")

            while self.state == GameState.ONGOING:
                chat_history = self.model(chat_history)
                guess = parse_guess(chat_history)
                if len(guess) != self.game.code_length:
                    # TODO: should we allow for retries if the model is too dumb to generate a 4 colors?
                    pass

                exact_matches, partial_matches, hint = self.game.evaluate(guess)
                current_score = (2 * exact_matches) + partial_matches
                if current_score > closest_score:
                    closest_score = current_score
                    closest_guess = guess

                self.attempts += 1
                progress_bar.update(1)

                if exact_matches == self.game.code_length:
                    self.state = GameState.WON
                    progress_bar.close()
                elif self.attempts >= self.game.max_guesses:
                    self.state = GameState.LOST
                    progress_bar.close()
                else:
                    chat_history.append({"role": "user", "content": f"Hint: {hint}.\nGuess: "})

            results.append(
                {
                    "chat_history": chat_history,
                    "solved": False if self.state == GameState.LOST else True,
                    "num_guesses": self.attempts,
                    "game": self.game.to_json(),
                    "model": self.model.get_model_info(),
                    "closest_score": closest_score,
                    "closest_guess": closest_guess,
                }
            )

        if save_results:
            if save_path is None:
                save_path = make_output_path()
            with open(save_path / "result.json", "w") as f:
                json.dump(results, f, indent=4)

        return results
