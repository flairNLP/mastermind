import json
import os
import string
from datasets import Dataset

from tqdm import tqdm
import random
from itertools import product
import inflect

p = inflect.engine()


def format_guesses_detail(guesses):
    formatted_guesses = []

    for guess, score in guesses:
        if score == [0, 0]:
            formatted_guesses.append(
                f"Guess: {guess}. Hint: none of the guessed colors are present in the secret code."
            )
        elif score[0] > 0 and score[1] == 0:
            formatted_guesses.append(
                f"Guess: {guess}. Hint: {p.number_to_words(score[0])} color{'s are' if score[0] > 1 else ' is'} in the correct position{'s' if score[0] > 1 else ''}."
            )
        elif score[0] == 0 and score[1] > 0:
            formatted_guesses.append(
                f"Guess: {guess}. Hint: {p.number_to_words(score[1])} color{'s are' if score[1] > 1 else ' is'} in the secret code but in the wrong position{'s' if score[1] > 1 else ''}."
            )
        elif score[0] > 0 and score[1] > 0:
            formatted_guesses.append(
                f"Guess: {guess}. Hint: {p.number_to_words(score[0])} color{'s are' if score[0] > 1 else ' is'} in the correct position{'s' if score[0] > 1 else ''} "
                f"and {p.number_to_words(score[1])} color{'s are' if score[1] > 1 else ' is'} in the secret code but in the wrong position{'s' if score[0] > 1 else ''}."
            )
        else:
            raise ValueError(f"Invalid feedback scores: {score}")

    return "\n".join(formatted_guesses)


def instruction_template(code_length: int, possible_colors: list, formatted_guesses: str):
    instruction = (
        f"Your goal is to find the secret {p.number_to_words(code_length)}-color code. The following colors are possible: {', '.join(possible_colors)}.\n"
        f"Some guesses have already been made. I will provide feedback for each guess made with which it is possible to unambigiously determine the secret code.\n\n"
        f"Previous Guesses:\n"
        f"{formatted_guesses}" + "\n\n"
    )
    return instruction


def generate_random_answers(code_length, possible_colors, already_guessed):
    possible_codes = [tuple(code) for code in product(possible_colors, repeat=code_length)]
    possible_codes = [code for code in possible_codes if code not in already_guessed]
    return random.sample(possible_codes, k=3)


def generate_close_answers(secret_code, possible_colors, guessed_options, num_tuples=3):
    """
    Generate close tuples based on a secret color code.
    A close tuple replaces one random entry in the secret code with a random color.
    Avoids previously guessed options.

    Args:
        secret_code (list): The secret color code (list of colors).
        possible_colors (list): List of all possible colors.
        guessed_options (list): List of already guessed tuples to avoid.
        num_tuples (int): Number of tuples to generate.

    Returns:
        list of tuples: List containing the generated close tuples.
    """
    close_tuples = set()  # Use a set to avoid duplicates

    while len(close_tuples) < num_tuples:
        # Copy the secret code to avoid modifying the original
        modified_code = secret_code.copy()

        # Randomly choose a position to replace
        position_to_replace = random.randint(0, len(secret_code) - 1)

        # Get a random color that is not the current color at that position
        original_color = modified_code[position_to_replace]
        new_color = random.choice([color for color in possible_colors if color != original_color])

        # Replace the color in the copied list
        modified_code[position_to_replace] = new_color

        # Convert the modified code to a tuple
        modified_tuple = tuple(modified_code)

        # Add to the result only if it is not already guessed and not in the result set
        if modified_tuple not in guessed_options and modified_tuple != tuple(secret_code):
            close_tuples.add(modified_tuple)

    return list(close_tuples)


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


def make_eval_harness(path: str):
    with open(f"{path}/raw.json") as f:
        data = json.load(f)

    dataset_easy = {"id": [], "instruction": [], "options": [], "answerKey": []}
    dataset_difficult = {"id": [], "instruction": [], "options": [], "answerKey": []}
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
        random_answer_options = generate_random_answers(
            code_length, possible_colors, [tuple(guess) for guess, _ in guesses]
        )
        close_answer_options = generate_close_answers(
            secret_code, possible_colors, [tuple(guess) for guess, _ in guesses]
        )
        random_answer_options, correct_random_option = prepare_shuffled_output(secret_code, random_answer_options)
        difficult_answer_options, correct_difficult_option = prepare_shuffled_output(secret_code, close_answer_options)
        formatted_hints = format_guesses_detail(hints)

        instruction = instruction_template(
            code_length=code_length,
            possible_colors=possible_colors,
            formatted_guesses=formatted_hints,
        )

        dataset_easy["id"].append(len(dataset_easy["id"]))
        dataset_easy["instruction"].append(instruction)
        dataset_easy["options"].append(random_answer_options)
        dataset_easy["answerKey"].append(correct_random_option)

        dataset_difficult["id"].append(len(dataset_difficult["id"]))
        dataset_difficult["instruction"].append(instruction)
        dataset_difficult["options"].append(difficult_answer_options)
        dataset_difficult["answerKey"].append(correct_difficult_option)

    print(f"Dataset (easy) contains {len(dataset_easy['id'])} instructions.")
    print(f"Dataset (difficult) contains {len(dataset_difficult['id'])} instructions.")

    with open(f"{path}/harness_easy.json", "w") as f:
        json.dump(dataset_easy, f, indent=4)

    with open(f"{path}/harness_difficult.json", "w") as f:
        json.dump(dataset_difficult, f, indent=4)

    return dataset_easy, dataset_difficult


if __name__ == "__main__":
    for setting in ["24", "35", "46"]:
        path = f"datasets/dataset_{setting}_50k"

        dataset_easy, dataset_difficult = make_eval_harness(path)
        dataset_easy = Dataset.from_dict(dataset_easy)
        easy_train_test = dataset_easy.train_test_split(test_size=0.05, seed=42)
        easy_train_val = easy_train_test["train"].train_test_split(test_size=0.1, seed=42)
        easy_train_val["validation"] = easy_train_val.pop("test")
        easy_train_val["test"] = easy_train_test["test"]
        easy_train_val.push_to_hub(f"flair/mastermind_{setting}_mcq_random")

        dataset_difficult = Dataset.from_dict(dataset_difficult)
        difficult_train_test = dataset_difficult.train_test_split(test_size=0.05, seed=42)
        difficult_train_val = difficult_train_test["train"].train_test_split(test_size=0.1, seed=42)
        difficult_train_val["validation"] = difficult_train_val.pop("test")
        difficult_train_val["test"] = difficult_train_test["test"]
        difficult_train_val.push_to_hub(f"flair/mastermind_{setting}_mcq_close")
