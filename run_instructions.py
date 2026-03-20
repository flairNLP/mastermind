import json
from argparse import ArgumentParser
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from mastermind.models import AnthropicModel, HFModel, OpenAIModel, VLLMModel
from mastermind.utils import parse_guess, make_output_path


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def evaluate(model, dataset, args):
    log = []

    for dp in tqdm(dataset, desc="Evaluating."):
        last_output = None
        try:
            last_output = model(dp["instruction"])
            guess = parse_guess(last_output[-1])
            log.append(
                {
                    "guess": guess,
                    "secret_code": dp["secret_code"],
                    "correct": guess == dp["secret_code"],
                    "valid": len(guess) == len(dp["secret_code"]),
                    "model": model.get_model_info(),
                    "dataset": args.dataset,
                }
            )
        except Exception as e:
            log.append(
                {
                    "guess": last_output[-1] if last_output else None,
                    "secret_code": dp["secret_code"],
                    "correct": False,
                    "valid": False,
                    "model": model.get_model_info(),
                    "dataset": args.dataset,
                }
            )

    output_path = make_output_path(game_type="deductive_reasoning")

    with open(output_path / "results.jsonl", "w") as f:
        for item in log:
            f.write(json.dumps(item) + "\n")

    with open(output_path / "info.json", "w") as f:
        json.dump({"model": model.get_model_info(), "dataset": args.dataset}, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hf")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", default="flair/mastermind_24_prompt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of shuffled training examples to evaluate.")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for compatible API backends such as vLLM.")
    parser.add_argument(
        "--enable_thinking",
        type=parse_optional_bool,
        default=None,
        help="Enable or disable thinking mode for supported chat templates such as Qwen3.",
    )
    arguments = parser.parse_args()

    dataset = load_dataset(arguments.dataset, split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(arguments.num_samples, len(dataset))))

    def apply_chat_template(example):
        example['instruction'] = [{"role": "user", "content": example['instruction']}]
        return example

    dataset = dataset.map(apply_chat_template)

    if arguments.model_type == "hf":
        model = HFModel(model_name=arguments.model, enable_thinking=arguments.enable_thinking)
        tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    elif arguments.model_type == "anthropic":
        model = AnthropicModel(model_name=arguments.model)
    elif arguments.model_type == "openai":
        model = OpenAIModel(model_name=arguments.model)
    elif arguments.model_type == "vllm":
        model = VLLMModel(model_name=arguments.model, base_url=arguments.base_url)
    else:
        raise ValueError(f"Invalid model type: {arguments.model_type}")

    evaluate(model, dataset, arguments)
