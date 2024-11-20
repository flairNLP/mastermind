from __future__ import division  # this ensures the / operation give floats by default
from __future__ import print_function

import json
from abc import ABC, abstractmethod
from typing import List, Tuple

import itertools
import time
from mastermind.evaluator import GameResult
from mastermind.game import Mastermind
from mastermind.utils import make_output_path


class Solver(ABC):
    def __init__(self, game: Mastermind):
        self.game = game

    @abstractmethod
    def solve(self, num_runs: int = 1, save_results: bool = False, save_path: str = "") -> List[GameResult]:
        raise NotImplementedError


class KnuthSolver(Solver):
    def __init__(self, game: Mastermind):
        super().__init__(game)

    def solve(self, num_runs: int = 1, save_results: bool = False, save_path: str = "") -> List[GameResult]:
        # init loggings
        results = []
        total_solved = 0
        total_guesses = 0
        worst_case = 0

        for _ in range(num_runs):
            # init guess is always half one colors half the other
            init_guess = init_guess = tuple(
                [1] * (self.game.code_length // 2) + [2] * (self.game.code_length - self.game.code_length // 2)
            )

            # solve the code
            guesses, process_time, progress = self.KnuthFive(init_guess=init_guess)

            if [self.game.possible_colors[i - 1] for i in list(guesses[-1])] == self.game.secret_code:
                total_solved += 1
                total_guesses += len(guesses)
                worst_case = max(worst_case, len(guesses))
                results.append(
                    {
                        'chat_history': None,
                        'valid': True,
                        'solved': True,
                        'num_guesses': len(guesses),
                        'precess_time': process_time,
                        'worst_case': worst_case,
                        "game": self.game.to_json(),
                        'model': None,
                        'progress_history': progress,
                    }
                )
            else:
                results.append(
                    {
                        'chat_history': None,
                        'valid': True,
                        'solved': False,
                        'precess_time': process_time,
                        'num_guesses': len(guesses),
                        "game": self.game.to_json(),
                        'model': None,
                        'progress_history': progress,
                    }
                )

            # reset the game
            self.game.reset()
        if save_results:
            results.append({'total_solved': total_solved, 'total_guesses': total_guesses})
            if save_path is None:
                save_path = str(make_output_path())
            with open(f"{save_path}/result.json", "w") as f:
                json.dump(results, f, indent=4)

        return results

    def evaluate_guess(self, guess):
        """
        This function compares a guess and a code and returns (numCorrect,numTransposed)
        which correspond to the number of black and white pegs returned in Mastermind.
        The code is a global variable, which the KnuthFive() function cannot use.
        """
        code = tuple(self.game.possible_colors.index(i) + 1 for i in self.game.secret_code)
        # Get the length n of the guess and the code.
        assert len(guess) == len(code)
        n = len(guess)

        # Determine the correct and incorrect positions.
        correct_positions = [i for i in list(range(n)) if guess[i] == code[i]]
        incorrect_positions = [i for i in list(range(n)) if guess[i] != code[i]]
        num_correct = len(correct_positions)

        # Reduce the guess and the code by removing the correct positions.
        # Create the set values that are common between the two reduced lists.
        reduced_guess = [guess[i] for i in incorrect_positions]
        reduced_code = [code[i] for i in incorrect_positions]
        reduced_set = set(reduced_guess) & set(reduced_guess)

        # Determine the number of transposed values.
        num_transposed = 0
        for x in reduced_set:
            num_transposed += min(reduced_guess.count(x), reduced_code.count(x))

        return num_correct, num_transposed

    def evaluation_inner(self, guess, fakeCode):
        """
        This function compares a guess and a temporary code and returns (numCorrect,numTransposed)
        which correspond to the number of black and white pegs returned in Mastermind.
        This function is created becuse actual code cannot be accessed in KnuthFive()
        """

        # Get the length n of the guess and the code.
        assert len(guess) == len(fakeCode)
        n = len(guess)

        # Determine the correct and incorrect positions.
        correct_positions = [i for i in list(range(n)) if guess[i] == fakeCode[i]]
        incorrect_positions = [i for i in list(range(n)) if guess[i] != fakeCode[i]]
        num_correct = len(correct_positions)

        # Reduce the guess and the fakeCode by removing the correct positions.
        # Create the set values that are common between the two reduced lists.
        reduced_guess = [guess[i] for i in incorrect_positions]
        reduced_code = [fakeCode[i] for i in incorrect_positions]
        reduced_set = set(reduced_guess) & set(reduced_guess)

        # Determine the number of transposed values.
        num_transposed = 0
        for x in reduced_set:
            num_transposed += min(reduced_guess.count(x), reduced_code.count(x))

        return num_correct, num_transposed

    def KnuthFive(self, init_guess: Tuple = (1, 1, 2, 2)):
        """
        Mastermind Knuth's five guess solver based on the implementation of catlzy (https://github.com/catlzy/knuth-mastermind/tree/master)

        Args:
            init_guess (Optional[Tuple]): The initial guess. For the default code_length of 4 the recommendet init_guess is (1,1,2,2)

        """
        start_time = time.time()
        progress = []
        # all the possible codes
        allP = [tuple(x) for x in itertools.product(range(1, self.game.num_colors + 1), repeat=self.game.code_length)]

        # the list contains all remaining possible solutions
        S = [tuple(x) for x in itertools.product(range(1, self.game.num_colors + 1), repeat=self.game.code_length)]

        # the following lines create a list (allScore) of all possible peg scores
        # allstemp is created because (3,1) is not allowed and thus is used for slicing later
        allstemp = []
        for i in range(self.game.code_length + 1):
            for j in range(0, self.game.code_length - i + 1):
                allstemp.append((i, j))
        allScore = allstemp[: len(allstemp) - 2] + allstemp[len(allstemp) - 1 :]

        # initial guess
        guess = init_guess
        guessList = [guess]
        guessName = [self.game.possible_colors[i - 1] for i in list(guess)]

        # num_correct, num_transposed = self.evaluate_guess(guess)
        # code = tuple(self.game.possible_colors.index(i) + 1 for i in self.game.secret_code)
        guesses = self.game.evaluate(guessName)
        num_correct = guesses[0]
        num_transposed = guesses[1]
        progress.append(guesses[2])

        # while the guess is not the code, keep guessing
        for _ in range(self.game.max_guesses):
            # temp is the list after removing all the conflicting guesses in S
            temp = []
            # cScore is a list of scores for each guess in allP
            cScore = [] * len(allP)

            # remove gueses from S that will not give the same peg score if it is the answer
            for item in S:
                if self.evaluation_inner(guess, item) == (num_correct, num_transposed):
                    temp.append(item)
            S = temp[:]

            # minimax step of the algorithm
            # for all unused guesses in allP, create a list (hitCount) that will be used
            # to calculate the score for the guess
            for item in allP:
                if item not in guessList:
                    hitCount = [0] * len(allScore)
                    # for all guesses in S, calculate its peg score if the unused guess in allP
                    # is the answer. Increase the corresponding position in hitCount by 1
                    for s in S:
                        hitCount[allScore.index(self.evaluation_inner(s, item))] += 1
                    # calculate the score for the current unused guess
                    cScore.append(len(S) - max(hitCount))
                else:
                    cScore.append(0)
            # find all indices with the max score
            maxScore = max(cScore)
            indices = [i for i, x in enumerate(cScore) if x == maxScore]
            # if any guesses corresponds to the indices is a member of S, use that as the next guess
            change = False
            for i in range(len(indices)):
                if allP[indices[i]] in S:
                    guess = allP[indices[i]]
                    guessName = [self.game.possible_colors[i - 1] for i in list(guess)]
                    change = True
                    break
            # else use the smallest guess as next guess
            if change is False:
                guess = allP[indices[0]]
                guessName = [self.game.possible_colors[i - 1] for i in list(guess)]
            guessList.append(guess)
            # num_correct, num_transposed = self.evaluate_guess(guess)
            guesses = self.game.evaluate(guessName)
            num_correct = guesses[0]
            num_transposed = guesses[1]
            progress.append(guesses[2])
            if (num_correct, num_transposed) == (self.game.code_length, 0):
                break
        return guessList, (time.time() - start_time), progress


if __name__ == "__main__":
    from mastermind.game import Mastermind

    game = Mastermind()
    solver = KnuthSolver(game)
    solver.solve(4, save_path='.', save_results=True)
