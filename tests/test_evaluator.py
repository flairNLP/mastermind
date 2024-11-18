from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind


def test_task_instruction_format(dummy_model):
    """Test task instruction format."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=False)
    evaluator = Evaluator(game, dummy_model)
    assert "Your task is to solve the game of Mastermind." in evaluator.task_instruction
    assert f"- You have to find out the {game.code_length}-color secret code." in evaluator.task_instruction
    assert f"- The following colors are allowed: {game.possible_colors}." in evaluator.task_instruction
    assert f"- Duplicates colors are not allowed." in evaluator.task_instruction
