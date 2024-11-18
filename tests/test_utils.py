from typing import List

import pytest

from mastermind.evaluator import parse_guess
from mastermind.models import CHAT_HISTORY


@pytest.mark.parametrize(
    "chat_history",
    [
        [{"role": "assistent", "content": "Guess: [green, yellow, red, blue]."}],
        [{"role": "assistent", "content": "Guess: ['green', 'yellow', 'red', 'blue']."}],
        [{"role": "assistent", "content": "Guess:[green, yellow, red, blue]."}],
        [{"role": "assistent", "content": "Based on previous results my answer is: [green, yellow, red, blue]."}],
        [
            {
                "role": "assistent",
                "content": "Based on previous results my answer is: Guess: [green, yellow, red, blue].",
            }
        ],
        [
            {
                "role": "assistent",
                "content": "Based on previous results my answer is: Guess: [green, green, green, green].\nGuess: [green, yellow, red, blue]",
            },
        ],
        [
            {
                "role": "assistent",
                "content": "Based on previous results my answer is: Guess: [green, green, green, green].\nGuess: [black, black, black, black]",
            },
            {
                "role": "assistent",
                "content": "Based on previous results my answer is: Guess: [green, green, green, green].\nGuess: [green, yellow, red, blue]",
            },
        ],
    ],
)
def test_parse_guess(chat_history: CHAT_HISTORY):
    assert parse_guess(chat_history) == ['green', 'yellow', 'red', 'blue']


@pytest.mark.parametrize(
    "chat_history",
    [
        [{"role": "assistent", "content": "[]"}],
        [{"role": "assistent", "content": ""}],
    ],
)
def test_parse_guess_invalid(chat_history: CHAT_HISTORY):
    assert parse_guess(chat_history) == []
