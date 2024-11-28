import random
from abc import ABC, abstractmethod
from itertools import product
from typing import List

from mastermind.game import Mastermind
from mastermind.models import ChatHistory


class Solver(ABC):
    def __init__(self, game: Mastermind):
        self.game = game

    @abstractmethod
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        raise NotImplementedError


class KnuthSolver(Solver):
    def __init__(self, game: Mastermind):
        super().__init__(game)
        self.initial_guess = random.sample(self.game.possible_colors, k=self.game.code_length)
        self.guesses = []
        self.all_possible_scores = self._compute_all_possible_scores(game)
        self.unused_guesses = [list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)]
        self.remaining_states = [
            list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)
        ]

    def _compute_all_possible_scores(self, game: Mastermind) -> List[List[int]]:
        all_possible_scores = [
            (black, white)
            for black in range(game.code_length + 1)  # `black` can range from 0 to `code_length`
            for white in range(game.code_length - black + 1)  # `white` depends on remaining slots
            if black + white <= game.code_length  # Total feedback must not exceed `code_length`
        ]
        return all_possible_scores

    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        current_guess = sum(["assistant" in x['role'] for x in chat_history])
        if current_guess == 0:
            self.guesses.append(self.initial_guess)
            chat_history.append({'role': 'assistant', 'content': self.initial_guess})
            self.unused_guesses.remove(self.initial_guess)
        else:
            guess = self._step()
            self.guesses.append(guess)
            chat_history.append({"role": "assistant", "content": guess})
            self.unused_guesses.remove(guess)
        return chat_history

    def _step(self) -> List[str]:
        last_guess = self.guesses[-1]
        exact_matches, partial_matches = self.game.evaluate_guess(last_guess, self.game.secret_code)

        temp = []
        for possible_next_guess in self.remaining_states:
            if self.game.evaluate_guess(last_guess, possible_next_guess) == (exact_matches, partial_matches):
                temp.append(possible_next_guess)
        self.remaining_states = temp[:]

        minimax_scores = [] * len(self.unused_guesses)
        for possible_next_guess in self.unused_guesses:
            hit_counter = [0] * len(self.all_possible_scores)
            # for all guesses in S, calculate its peg score if the unused guess in allP
            # is the answer. Increase the corresponding position in hitCount by 1
            for logical_guess in self.remaining_states:
                hit_counter[
                    self.all_possible_scores.index(self.game.evaluate_guess(logical_guess, possible_next_guess))
                ] += 1
            # calculate the score for the current unused guess
            minimax_scores.append(len(self.remaining_states) - max(hit_counter))
        # find all indices with the max score
        max_score = max(minimax_scores)
        indices = [i for i, x in enumerate(minimax_scores) if x == max_score]
        # if any guesses corresponds to the indices is a member of S, use that as the next guess
        change = False
        for i in range(len(indices)):
            if self.unused_guesses[indices[i]] in self.remaining_states:
                guess = self.unused_guesses[indices[i]]
                change = True
                break
        # else use the smallest guess as next guess
        if change is False:
            guess = self.unused_guesses[indices[0]]
        self.guesses.append(guess)
        return guess

    def get_model_info(self) -> str:
        return "KnuthSolver"

    def reset(self):
        self.guesses = []
        self.unused_guesses = [list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)]
        self.remaining_states = [
            list(code) for code in product(self.game.possible_colors, repeat=self.game.code_length)
        ]
        self.initial_guess = random.sample(self.game.possible_colors, k=self.game.code_length)
