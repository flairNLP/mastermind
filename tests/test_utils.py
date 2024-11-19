import pytest

from mastermind.evaluator import parse_guess
from mastermind.models import ChatHistory
from mastermind.utils import make_output_path


@pytest.mark.parametrize(
    "chat_history",
    [
        [{"role": "assistent", "content": "[green, yellow, red, blue]."}],
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
def test_parse_guess(chat_history: ChatHistory):
    assert parse_guess(chat_history) == ['green', 'yellow', 'red', 'blue']


@pytest.mark.parametrize(
    "chat_history",
    [
        [{"role": "assistent", "content": "[]"}],
        [{"role": "assistent", "content": ""}],
    ],
)
def test_parse_guess_invalid(chat_history: ChatHistory):
    assert parse_guess(chat_history) == []


def test_make_output_path(tmp_path):
    # Call the function
    output_path = make_output_path(tmp_path)

    # Check if the directory is created correctly
    assert output_path.exists()
    assert tmp_path in output_path.parents
