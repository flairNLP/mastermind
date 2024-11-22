import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import rootutils

from mastermind.models import ChatHistory

COLOR_MAP = {
    "red": "\033[91m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "orange": "\033[33m",
    "purple": "\033[95m",
    "pink": "\033[95m",
    "brown": "\033[33m",
    "black": "\033[90m",
    "white": "\033[97m",
}
RESET = "\033[0m"


def make_output_path(base_path: Optional[Path] = None) -> Path:
    if not base_path:
        base_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(base_path / f"results/{current_datetime}")
    output_path.mkdir(parents=True)
    return output_path


def parse_guess(chat_history: ChatHistory) -> List[str]:
    if isinstance(chat_history[-1]['content'], list):
        return chat_history[-1]['content']
    # Regular expression to extract content within "Guess: [ ... ]"
    elif isinstance(chat_history[-1]['content'], str):
        matches = re.findall(r"(?:Guess:\s*)?\[([^\]]+)\]", chat_history[-1]["content"])
        if matches:
            # Split the matched content into a list of strings
            return [item.strip().strip("'").strip('"') for item in matches[-1].split(",")]
        else:
            return []
    else:
        return []


def colorize_code(code):
    return [f"{COLOR_MAP.get(color, '')}{color}{RESET}" for color in code]


def print_summary(model, game, args, result):
    # Calculate padding for alignment
    labels = [
        "Model:",
        "Code Length:",
        "Number of Colors:",
        "Last Code:",
        "Duplicates Allowed:",
        "Max Guesses:",
        "Games Played:",
        "Games Won:",
    ]
    max_label_length = max(len(label) for label in labels)

    # Text output
    print("=== Summary ===")
    print(f"{'Model:'.ljust(max_label_length)} {model.get_model_info()}")
    print(f"{'Code Length:'.ljust(max_label_length)} {game.code_length}")
    print(f"{'Number of Colors:'.ljust(max_label_length)} {game.num_colors}")
    print(f"{'Last Code:'.ljust(max_label_length)} {', '.join(colorize_code(game.secret_code))}")
    print(f"{'Duplicates Allowed:'.ljust(max_label_length)} {game.duplicates_allowed}")
    print(f"{'Max Guesses:'.ljust(max_label_length)} {game.max_guesses}")
    print(f"{'Games Played:'.ljust(max_label_length)} {args.num_runs}")
    print(f"{'Games Won:'.ljust(max_label_length)} {sum(r['solved'] for r in result)}")
