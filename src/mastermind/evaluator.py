import json
import random
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from mastermind.game import Mastermind
from mastermind.models import ChatHistory, LanguageModel
from mastermind.solvers import Solver
from mastermind.utils import COLOR_MAP, RESET, make_output_path, parse_guess

GameResult = Dict[str, str]
Guess = List[str]
Scores = Tuple[int, int]
GuessHistory = List[Tuple[Guess, Scores]]
ProgressHistory = List[int]

GREEN = COLOR_MAP["green"]
RED = COLOR_MAP["red"]
YELLOW = COLOR_MAP["yellow"]


class GameState(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Evaluator:
    def __init__(
        self,
        game: Mastermind,
        model: Union[LanguageModel, Solver],
        use_cot: bool = False,
        use_fewshot_example: bool = False,
        compute_progress: bool = False,
    ):
        self.game = game
        self.model = model
        self.attempts = 0
        self.state = GameState.ONGOING
        self.use_cot = use_cot
        self.use_fewshot_example = use_fewshot_example
        self.compute_progress = compute_progress

    def _init_chat_history(self) -> ChatHistory:
        return [
            {"role": "user", "content": f"{self._init_instruction()}\n\n{self._guess_template()}"},
        ]

    def _init_instruction(self) -> str:
        task_instruction = (
            f"Your task is to solve the game of Mastermind. The game is defined as follows:\n"
            f"- You have to find out the {self.game.code_length}-color secret code.\n"
            f"- The following colors are allowed: {self.game.possible_colors}.\n"
            f"- Duplicate colors in a single code are allowed.\n"
            f"- You have a total of {self.game.max_guesses} guesses.\n"
            "- After each guess, I will provide feedback on:\n"
            "  1. How many guesses are the correct color and in the correct position.\n"
            "  2. How many guesses are the correct color but in the wrong position.\n"
            "- Consider the feedback from all your guesses to find out the secret code.\n"
            "- You can must clearly indicate your final answer by preprending 'FINAL GUESS:' to it.\n\n"
            f"{self._example_template()}"
        )
        return task_instruction

    def _guess_template(self) -> str:
        return f"{'Before giving your next guess, analyze your next guess using all previous feedback step-by-step. ' if self.use_cot else ''}What's your next guess?"

    def _example_template(self) -> str:
        if self.use_fewshot_example:
            example = (
                "###"
                "Full Game Example (Secret Code: ['white', 'green', 'red', 'brown']):\n"
                "<User> What's your next guess?\n"
                "<Assistant> FINAL GUESS:['yellow', 'black', 'green', 'brown']\n"
                "<User> Feedback: Correct color and position: 1. Correct color but wrong position: 1.\nWhat's your next guess?"
                "<Assistant> FINAL GUESS:['white', 'brown', 'green', 'brown']\n"
                "<User> Feedback: Correct color and position: 2. Correct color but wrong position: 1.\nWhat's your next guess?"
                "<Assistant> FINAL GUESS:['white', 'green', 'white', 'brown']\n"
                "<User> Feedback: Correct color and position: 3. Correct color but wrong position: 0.\nWhat's your next guess?"
                "<Assistant> FINAL GUESS:['white', 'green', 'red', 'brown']\n"
                "<User> Feedback: Correct color and position: 4. Correct color but wrong position: 0.\nYou solved it!"
                "###"
            )
        else:
            example = (
                "### Example:\n"
                "<User> What's your next guess?\n"
                f"<Assistant> FINAL GUESS:{random.sample(self.game.possible_colors, k=self.game.code_length)}\n"
                f"<User> Feedback: <number> color(s) in the correct position(s). <number> color(s) but wrong position(s).\n{self._guess_template()}\n"
                "###"
            )

        return example

    def run(
        self,
        num_games: int = 1,
        save_results: bool = False,
        save_path: Optional[Path] = None,
        compute_progress: bool = False,
    ) -> List[GameResult]:
        results = []
        for num_game in range(num_games):
            chat_history: ChatHistory = self._init_chat_history()
            guess_history: GuessHistory = []
            total_guesses_bar = tqdm(
                total=self.game.max_guesses, desc=f"{YELLOW}[Game #{num_game}]{RESET} Attempts", unit="attempt"
            )

            try:
                while self.state == GameState.ONGOING:
                    chat_history = self.model(chat_history)
                    guess = parse_guess(chat_history[-1])
                    exact_matches, partial_matches, hint = self.game.evaluate(guess)
                    guess_history.append((guess, (exact_matches, partial_matches)))

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
                        chat_history.append({"role": "user", "content": f"Feedback: {hint}\n{self._guess_template()}"})

                if compute_progress:
                    progress_history = self.progress(guess_history)

                result = {
                    "chat_history": chat_history,
                    "guess_history": guess_history,
                    "progress_history": progress_history,
                    "valid": True if not self.attempts == 1 else False,  # random guesses are not valid
                    "solved": False if self.state == GameState.LOST else True,
                    "num_guesses": self.attempts,
                    "game": self.game.to_json(),
                    "model": self.model.get_model_info(),
                }

                if save_results:
                    if save_path is None:
                        save_path = make_output_path()
                    with open(save_path / "results.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")

            except Exception as e:
                print(f"Error occurred: {e}")

            self.reset()

        return results

    def progress(self, guess_history: GuessHistory) -> ProgressHistory:
        all_codes = list(product(self.game.possible_colors, repeat=self.game.code_length))
        progress_history = [len(all_codes)]
        for i in range(1, len(guess_history)):
            guesses = guess_history[:i]
            remaining_states = []
            for possible_code in all_codes:
                if all(self.game.evaluate_guess(guess, possible_code) == feedback for guess, feedback in guesses):
                    remaining_states.append(possible_code)
            progress_history.append(len(remaining_states))
        return progress_history

    def reset(self):
        self.game.reset()
        self.state = GameState.ONGOING
        self.attempts = 0
        if isinstance(self.model, Solver):
            self.model.reset()
