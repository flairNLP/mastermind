import json
import random
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from mastermind.game import Mastermind
from mastermind.models import ChatHistory, LanguageModel
from mastermind.solvers import Solver
from mastermind.utils import COLOR_MAP, RESET, make_output_path, parse_guess

GameResult = Dict[str, str]
Progress = Dict[str, bool]

GREEN = COLOR_MAP["green"]
RED = COLOR_MAP["red"]
YELLOW = COLOR_MAP["yellow"]


class GameState(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Evaluator:
    def __init__(self, game: Mastermind, model: Union[LanguageModel, Solver], compute_progress: bool = False):
        self.game = game
        self.model = model
        self.attempts = 0
        self.state = GameState.ONGOING
        self.compute_progress = compute_progress

    def _init_progress_attributes(self):
        self.all_possible_scores = self.game.compute_all_possible_scores()
        self.unused_guesses = [list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)]
        self.remaining_states = [
            list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)
        ]
        self.optimal_next_guesses = [
            list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)
        ]

    def _init_chat_history(self) -> ChatHistory:
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

    def run(
        self,
        num_games: int = 1,
        save_results: bool = False,
        save_path: Optional[Path] = None,
        compute_progress: bool = False,
    ) -> List[GameResult]:
        results = []
        for num_game in range(num_games):
            chat_history = self._init_chat_history()
            if compute_progress:
                self._init_progress_attributes()
                progress_history = []
            total_guesses_bar = tqdm(
                total=self.game.max_guesses, desc=f"{YELLOW}[Game #{num_game}]{RESET} Attempts", unit="attempt"
            )
            try:
                while self.state == GameState.ONGOING:
                    chat_history = self.model(chat_history)
                    guess = parse_guess(chat_history)

                    exact_matches, partial_matches, hint = self.game.evaluate(guess)
                    if compute_progress:
                        progress = self.progress(guess, exact_matches, partial_matches)
                        progress_history.append(progress)

                    self.attempts += 1
                    total_guesses_bar.update(1)

                    if exact_matches == self.game.code_length:
                        total_guesses_bar.desc = f"{GREEN}[Game #{num_game}] Game Solved{RESET}"
                        total_guesses_bar.refresh()
                        self.state = GameState.WON
                        total_guesses_bar.close()
                    elif self.attempts >= self.game.max_guesses:
                        total_guesses_bar.desc = f"{RED}[Game #{num_game}] Game Over{RESET}"
                        total_guesses_bar.refresh()
                        self.state = GameState.LOST
                        total_guesses_bar.close()
                    else:
                        chat_history.append({"role": "user", "content": f"Feedback: {hint}\nGuess: "})

                results.append(
                    {
                        "chat_history": chat_history,
                        "valid": True,
                        "solved": False if self.state == GameState.LOST else True,
                        "num_guesses": self.attempts,
                        "game": self.game.to_json(),
                        "model": self.model.get_model_info(),
                        "progress_history": progress_history,
                    }
                )

            except Exception as e:
                print(f"Error occurred: {e}")
                results.append(
                    {
                        "chat_history": chat_history,
                        "valid": False,
                        "solved": False,
                        "num_guesses": self.attempts,
                        "game": self.game.to_json(),
                        "model": self.model.get_model_info(),
                        "progress_history": progress_history,
                        "error": str(e),
                    }
                )

            self.reset()

        if save_results:
            if save_path is None:
                save_path = make_output_path()
            with open(save_path / "result.json", "w") as f:
                json.dump(results, f, indent=4)

        return results

    def progress(self, guess: List[str], exact_matches: int, partial_matches: int) -> Progress:
        self.unused_guesses.remove(guess)
        is_logical_guess = guess in self.remaining_states
        self.compute_remaining_states(guess, exact_matches, partial_matches)
        is_optimal_guess = guess in self.optimal_next_guesses
        self.compute_optimal_next_guesses()
        return {"is_logical_guess": is_logical_guess, "is_optimal_guess": is_optimal_guess}

    def compute_remaining_states(self, current_guess: List[str], exact_matches: int, partial_matches: int):
        tmp = []
        for possible_code in self.remaining_states:
            if self.game.evaluate_guess(current_guess, possible_code) == (exact_matches, partial_matches):
                tmp.append(possible_code)
        self.remaining_states = tmp[:]

    def compute_optimal_next_guesses(self):
        minimax_scores = []
        for item in self.unused_guesses:
            hit_count = [0] * len(self.all_possible_scores)
            for possible_code in self.remaining_states:
                hit_count[self.all_possible_scores.index(self.game.evaluate_guess(possible_code, item))] += 1
            minimax_scores.append(len(self.remaining_states) - max(hit_count))

        max_score = max(minimax_scores)
        indices = [i for i, x in enumerate(minimax_scores) if x == max_score]
        optimal_next_guesses = [self.unused_guesses[i] for i in indices]
        self.optimal_next_guesses = optimal_next_guesses

    def reset(self):
        self.game.reset()
        self.state = GameState.ONGOING
        self.attempts = 0
        if isinstance(self.model, Solver):
            self.model.reset()
