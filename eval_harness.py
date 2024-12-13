import json
import os
import string
from argparse import ArgumentParser
from datasets import Dataset, load_dataset

from tqdm import tqdm
import random
from itertools import product


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


def instruction_template(code_length: int, possible_colors: list, formatted_hints: str):
    instruction = (
        f"You are playing the game Mastermind, and your goal is to find the secret {code_length}-color code. The following colors are possible: {', '.join(possible_colors)}. In total, there are {code_length ** len(possible_colors)} possible codes.\n"
        f"I will provide a series of guesses along with their respective feedback. The feedback will logically eliminate all possibilities except for the correct code.\n"
        "The feedback specifies:\n"
        "1. The number of colors that are both present in the secret code and in the correct position.\n"
        "2. The number of colors that are present in the secret code but in the wrong position, excluding those already counted in the correct position.\n"
        "3. If a color appears multiple times, its occurrences are matched proportionally.\n\n"
        f"Below are the guesses and their feedback:\n"
        f"{formatted_hints}" + "\n"
    )
    return instruction


def random_answers(code_length, possible_colors, already_guessed):
    possible_codes = [tuple(code) for code in product(possible_colors, repeat=code_length)]
    possible_codes = [code for code in possible_codes if code not in already_guessed]
    return random.sample(possible_codes, k=3)


def prepare_shuffled_output(secret, options):
    # Combine secret with options and shuffle
    options = [", ".join(option) for option in options]
    secret = ", ".join(secret)
    all_options = options + [secret]
    random.shuffle(all_options)

    # Assign labels (A, B, C, ...) to the options
    labels = list(string.ascii_uppercase[: len(all_options)])
    label_mapping = {option: label for option, label in zip(all_options, labels)}

    # Find the correct label for the secret
    correct_label = label_mapping[secret]

    # Prepare the final output
    output = {"text": all_options, "label": labels}

    return output, correct_label


def make_eval_harness(args):
    if os.path.exists(f"{args.dataset}/mastermind_eval_harness.json"):
        with open(f"{args.dataset}/mastermind_eval_harness.json") as f:
            dataset = json.load(f)
        return dataset

    with open(f"{args.dataset}/raw.json") as f:
        data = json.load(f)

    dataset = {"id": [], "instruction": [], "options": [], "answerKey": []}
    for game in tqdm(data, desc="Filtering data."):
        progress_history = game["progress_history"]
        if not progress_history[-1] == 1:
            continue

        game_data = game["game"]
        guesses = game["guess_history"]
        hints = game["guess_history"][:-1]
        secret_code = game_data["secret_code"]
        possible_colors = game_data["possible_colors"]
        code_length = game_data["code_length"]
        wrong_answer_options = random_answers(code_length, possible_colors, [tuple(guess) for guess, _ in guesses])
        answer_options, correct_option = prepare_shuffled_output(secret_code, wrong_answer_options)
        formatted_hints = format_guesses_detail(hints)

        instruction = instruction_template(
            code_length=code_length,
            possible_colors=possible_colors,
            formatted_hints=formatted_hints,
        )

        dataset["id"].append(len(dataset["id"]))
        dataset["instruction"].append(instruction)
        dataset["options"].append(answer_options)
        dataset["answerKey"].append(correct_option)

    print(f"Dataset contains {len(dataset['id'])} instructions.")

    with open(f"{args.dataset}/mastermind_eval_harness.json", "w") as f:
        json.dump(dataset, f, indent=4)

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset_46_50k")
    arguments = parser.parse_args()

    dataset_raw = make_eval_harness(arguments)
    dataset = Dataset.from_dict(dataset_raw)
    dataset.push_to_hub("mastermind_46")
