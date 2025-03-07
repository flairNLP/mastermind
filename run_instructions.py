import json
from argparse import ArgumentParser

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from mastermind.models import AnthropicModel, HFModel, OpenAIModel
from mastermind.utils import parse_guess, make_output_path


def evaluate(model, dataset, args):
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

    output_path = make_output_path(game_type="instructions")

    with open(output_path / "results.jsonl", "w") as f:
        for item in log:
            f.write(json.dumps(item) + "\n")

    with open(output_path / "info.json", "w") as f:
        json.dump({"model": model.get_model_info(), "dataset": args.dataset}, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default="flair/mastermind_24_prompt")
    arguments = parser.parse_args()

    dataset = load_dataset(arguments.dataset, split="train")
    dataset = dataset.shuffle(seed=42).select(range(100))

    def apply_chat_template(example):
        example['instruction'] = [{"role": "user", "content": example['instruction']}]
        return example

    dataset = dataset.map(apply_chat_template)

    if arguments.model_type == "hf":
        model = HFModel(model_name=arguments.model)
        tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    elif arguments.model_type == "anthropic":
        model = AnthropicModel(model_name=arguments.model)
    elif arguments.model_type == "openai":
        model = OpenAIModel(model_name=arguments.model)
    else:
        raise ValueError(f"Invalid model type: {arguments.model_type}")

    evaluate(model, dataset, num_runs=100)
