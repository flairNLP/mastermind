import json
import random
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from mastermind.game import Mastermind
from mastermind.models import ChatHistory, LanguageModel
from mastermind.utils import make_output_path, parse_guess

GameResult = Dict[str, str]


class GameState(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Evaluator:
    def __init__(self, game: Mastermind, model: LanguageModel):
        self.game = game
        self.model = model
        self.attempts = 0
        self.state = GameState.ONGOING

    def init_chat_history(self) -> ChatHistory:
        task_instruction = (
            f"Your task is to solve the game of Mastermind. The game is defined as follows:\n"
            f"- You have to find out the {self.game.code_length}-color secret code.\n"
            f"- The following colors are allowed: {self.game.possible_colors}.\n"
            f"- Duplicates are{' ' if self.game.duplicates_allowed else ' not '}allowed.\n"
            f"- You have a total of {self.game.max_guesses} guesses.\n"
            "- After each guess, I will provide feedback on:\n"
            "  1. How many guesses are the correct color and in the correct position.\n"
            "  2. How many guesses are the correct color but in the wrong position.\n\n"
            "We interact like in the following example:\n###\n"
            f"Guess: {random.sample(self.game.possible_colors, k=self.game.code_length)}\n\n"
            "Feedback: Correct color and position: <number>. Correct color but wrong position: <number>.\n"
            "###\n\n"
        )
        return [{"role": "user", "content": task_instruction}, {"role": "user", "content": "Guess: "}]

    def run(self, num_games: int = 1, save_results: bool = False, save_path: Optional[Path] = None) -> List[GameResult]:
        results = []
        for _ in range(num_games):
            chat_history = self.init_chat_history()
            progress_history = []
            total_guesses_bar = tqdm(total=self.game.max_guesses, desc="Attempts", unit="attempt")

            while self.state == GameState.ONGOING:
                chat_history = self.model(chat_history)
                guess = parse_guess(chat_history)
                if len(guess) != self.game.code_length:
                    # TODO: should we allow for retries if the model is too dumb to generate a 4 colors?
                    pass

                exact_matches, _, progress, hint = self.game.evaluate(guess)
                progress_history.append(progress)

                self.attempts += 1
                total_guesses_bar.update(1)

                if exact_matches == self.game.code_length:
                    self.state = GameState.WON
                    total_guesses_bar.close()
                elif self.attempts >= self.game.max_guesses:
                    self.state = GameState.LOST
                    total_guesses_bar.close()
                else:
                    chat_history.append({"role": "user", "content": f"Feedback: {hint}.\nGuess: "})

            results.append(
                {
                    "chat_history": chat_history,
                    "solved": False if self.state == GameState.LOST else True,
                    "num_guesses": self.attempts,
                    "game": self.game.to_json(),
                    "model": self.model.get_model_info(),
                    "progress_history": progress_history,
                }
            )
            self.reset()

        if save_results:
            if save_path is None:
                save_path = make_output_path()
            with open(save_path / "result.json", "w") as f:
                json.dump(results, f, indent=4)

        return results

    def reset(self):
        self.game.reset()
        self.state = GameState.ONGOING
        self.attempts = 0
