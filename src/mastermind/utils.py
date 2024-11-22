import re
from datetime import datetime
from pathlib import Path
from typing import List

import rootutils

from mastermind.models import ChatHistory


def make_output_path(base_path: Path = None) -> Path:
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
