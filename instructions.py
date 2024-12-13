import json
import os
from argparse import ArgumentParser

from tqdm import tqdm
import random

from mastermind.models import AnthropicModel, HFModel, OpenAIModel
from mastermind.utils import parse_guess, make_output_path


def format_guesses_plain(guesses):
    formatted_guesses = []

    for guess, score in guesses:
        formatted_guesses.append(
            f"Guess: {guess} -> "
            f"Correct colors in the correct positions: {score[0]}. "
            f"Correct colors in the wrong positions: {score[1]}."
        )

    return "\n".join(formatted_guesses)


def format_guesses_detail(guesses):
    formatted_guesses = []

    for guess, score in guesses:
        if score == [0, 0]:
            formatted_guesses.append(f"Guess: {guess} -> None of the guessed colors are present in the secret code.")
        elif score[0] > 0 and score[1] == 0:
            formatted_guesses.append(
                f"Guess: {guess} -> {score[0]} color(s) is/are correct and in the correct position(s)."
            )
        elif score[0] == 0 and score[1] > 0:
            formatted_guesses.append(
                f"Guess: {guess} -> {score[1]} color(s) is/are present in the secret code but in the wrong position(s)."
            )
        elif score[0] > 0 and score[1] > 0:
            formatted_guesses.append(
                f"Guess: {guess} ->  {score[0]} color(s) is/are correct and in the correct position(s), "
                f"and {score[1]} color(s) is/are present in the secret code but in the wrong position(s)."
            )
        else:
            raise ValueError(f"Invalid feedback scores: {score}")

    return "\n".join(formatted_guesses)


def instruction_template(code_length: int, possible_colors: list, formatted_guesses: str):
    instruction = (
        f"You are playing the game Mastermind, and your goal is to find the secret {code_length}-color code. The following colors are possible: {', '.join(possible_colors)}. In total, there are {code_length ** len(possible_colors)} possible codes.\n"
        f"I will provide a series of guesses along with their respective feedback. The feedback will logically eliminate all possibilities except for the correct code.\n"
        "The feedback specifies:\n"
        "1. The number of colors that are both present in the secret code and in the correct position.\n"
        "2. The number of colors that are present in the secret code but in the wrong position, excluding those already counted in the correct position.\n"
        "3. If a color appears multiple times, its occurrences are matched proportionally.\n\n"
        f"Below are the guesses and their feedback:\n"
        f"{formatted_guesses}" + "\n"
        f"Using these hints, deduce the one remaining {code_length}-color code, which is the secret code.\n"
        "Explain in detail how you deduce the secret code and clearly indicate your final guess at the end, starting it with 'FINAL GUESS:'."
    )
    return instruction


def clean(args):
    if os.path.exists(f"{args.dataset}/instructions.jsonl"):
        print(f"Dataset already exists at {args.dataset}.")
        return

    with open(f"{args.dataset}/raw.json") as f:
        data = json.load(f)

    instructions = []
    for game in tqdm(data, desc="Filtering data."):
        progress_history = game["progress_history"]
        if not progress_history[-1] == 1:
            continue

        game_data = game["game"]
        guesses = game["guess_history"][:-1]
        formatted_guesses = format_guesses_detail(guesses)

        instruction = instruction_template(
            code_length=game_data["code_length"],
            possible_colors=game_data["possible_colors"],
            formatted_guesses=formatted_guesses,
        )

        instructions.append(
            {"instruction": [{"content": instruction, "role": "user"}], "secret_code": game_data['secret_code']}
        )

    print(f"Dataset contains {len(instructions)} instructions.")

    with open(f"{args.dataset}/instructions.jsonl", "w") as f:
        for instruction in instructions:
            f.write(json.dumps(instruction) + "\n")


def load(args, k=500):
    instructions = []
    with open(f"{args.dataset}/instructions.jsonl") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading instructions"):
            instructions.append(json.loads(line))
    return random.sample(instructions, k)


def test(model, dataset, args):
    log = []

    for dp in tqdm(dataset, desc="Evaluating."):
        try:
            out = model(dp["instruction"])
            guess = parse_guess(out[-1])
            log.append(
                {
                    "guess": guess,
                    "secret_code": dp["secret_code"],
                    "correct": guess == dp["secret_code"],
                    "valid": len(guess) == len(dp["secret_code"]),
                }
            )
        except Exception as e:
            log.append({"guess": out[-1], "secret_code": dp["secret_code"], "correct": False, "valid": False})

    output_path = make_output_path()
    with open(output_path / "results.json", "w") as f:
        json.dump(log, f)

    with open(output_path / "info.json", "w") as f:
        json.dump({"model": model.get_model_info(), "dataset": args.dataset}, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default="datasets/dataset_24_50k")
    arguments = parser.parse_args()

    clean(arguments)
    random.seed(42)
    dataset = load(arguments)

    if arguments.model_type == "hf":
        model = HFModel(model_name=arguments.model)
    elif arguments.model_type == "anthropic":
        model = AnthropicModel(model_name=arguments.model)
    elif arguments.model_type == "openai":
        model = OpenAIModel(model_name=arguments.model)
    else:
        raise ValueError(f"Invalid model type: {arguments.model_type}")

    test(model, dataset, arguments)
