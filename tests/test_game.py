import pytest

from mastermind import COLORS, Mastermind


def test_initialization():
    """Test initialization of the Mastermind class."""
    game = Mastermind(code_length=4, num_colors=6, max_guesses=10, duplicates_allowed=True)

    # Check basic properties
    assert game.code_length == 4
    assert game.num_colors == 6
    assert game.max_guesses == 10
    assert game.duplicates_allowed is True

    # Check possible colors
    assert len(game.possible_colors) == 6
    assert all(color in COLORS for color in game.possible_colors)

    # Check secret code generation
    assert len(game.secret_code) == 4
    if not game.duplicates_allowed:
        assert len(set(game.secret_code)) == len(game.secret_code)


def test_generate_secret_code_no_duplicates():
    """Test secret code generation without duplicates."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=False)
    assert len(game.secret_code) == 4
    assert len(set(game.secret_code)) == 4  # Ensure no duplicates


def test_generate_secret_code_with_duplicates():
    """Test secret code generation with duplicates."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=True)
    assert len(game.secret_code) == 4  # Length matches code_length


def test_evaluate_guess_exact_matches():
    """Test exact match evaluation."""
    game = Mastermind(code_length=4, num_colors=4, duplicates_allowed=False)
    game.secret_code = ["red", "blue", "green", "yellow"]  # Mock secret code
    guess = ["red", "blue", "green", "yellow"]
    exact_matches, partial_matches = game.evaluate_guess(guess)
    assert exact_matches == 4
    assert partial_matches == 0


def test_evaluate_guess_partial_matches():
    """Test partial match evaluation."""
    game = Mastermind(code_length=4, num_colors=4, duplicates_allowed=False)
    game.secret_code = ["red", "blue", "green", "yellow"]  # Mock secret code
    guess = ["blue", "red", "yellow", "green"]
    exact_matches, partial_matches = game.evaluate_guess(guess)
    assert exact_matches == 0
    assert partial_matches == 4


def test_evaluate_guess_mixed_matches():
    """Test mixed exact and partial matches."""
    game = Mastermind(code_length=4, num_colors=4, duplicates_allowed=False)
    game.secret_code = ["red", "blue", "green", "yellow"]  # Mock secret code
    guess = ["red", "yellow", "green", "blue"]
    exact_matches, partial_matches = game.evaluate_guess(guess)
    assert exact_matches == 2  # "red" and "green" in the correct position
    assert partial_matches == 2  # "yellow" and "blue" in the wrong position


def test_invalid_guess_length():
    """Test invalid guess length."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=True)
    with pytest.raises(ValueError, match="Guess must be of length 4."):
        game.evaluate(["red", "blue"])  # Guess is too short


def test_evaluate_with_hint():
    """Test evaluation with hint."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=True)
    game.secret_code = ["red", "blue", "green", "yellow"]  # Mock secret code
    guess = ["red", "yellow", "green", "blue"]
    exact_matches, partial_matches, hint = game.evaluate(guess)
    assert exact_matches == 2
    assert partial_matches == 2
    assert hint == "Correct color and position: 2. Correct color but wrong position: 2."


def test_repr():
    """Test the __repr__ method."""
    game = Mastermind(code_length=4, num_colors=6, duplicates_allowed=True)
    repr_str = repr(game)
    assert "Mastermind" in repr_str
    assert "code_length=4" in repr_str
    assert "possible_colors=6" in repr_str
    assert "secret_code_hidden=True" in repr_str
