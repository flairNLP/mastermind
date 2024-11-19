from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind


def test_task_instruction_format(dummy_model):
    """Test task instruction format."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=False)
    evaluator = Evaluator(game, dummy_model)
    chat_history = evaluator.init_chat_history()
    assert "Your task is to solve the game of Mastermind." in chat_history[0]["content"]
    assert f"- You have to find out the {game.code_length}-color secret code." in chat_history[0]["content"]
    assert f"- The following colors are allowed: {game.possible_colors}." in chat_history[0]["content"]
    assert f"- Duplicates are not allowed." in chat_history[0]["content"]
    assert "Guess:" in chat_history[1]["content"]
