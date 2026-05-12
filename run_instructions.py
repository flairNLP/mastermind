import json
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm
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
    log_lock = threading.Lock()

    pbar = tqdm(total=len(dataset), desc="Evaluating.")

    def process(dp):
        last_output = None
        try:
            last_output = model(dp["instruction"])
            guess = parse_guess(last_output[-1])
            return {
                "guess": guess,
                "secret_code": dp["secret_code"],
                "correct": guess == dp["secret_code"],
                "valid": len(guess) == len(dp["secret_code"]),
                "model": model.get_model_info(),
                "dataset": args.dataset,
            }
        except Exception:
            return {
                "guess": last_output[-1] if last_output else None,
                "secret_code": dp["secret_code"],
                "correct": False,
                "valid": False,
                "model": model.get_model_info(),
                "dataset": args.dataset,
            }

    with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
        futures = {executor.submit(process, dp): dp for dp in dataset}
        for future in as_completed(futures):
            result = future.result()
            with log_lock:
                log.append(result)
            pbar.update(1)

    pbar.close()

    output_path = make_output_path(base_path=args.save_path, game_type="deductive_reasoning")

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
    parser.add_argument("--save_path", type=str, default=None, help="Base directory for saving results.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of samples to evaluate in parallel (useful with vLLM).")
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
    elif arguments.model_type == "anthropic":
        model = AnthropicModel(model_name=arguments.model)
    elif arguments.model_type == "openai":
        model = OpenAIModel(model_name=arguments.model)
    elif arguments.model_type == "vllm":
        model = VLLMModel(model_name=arguments.model, base_url=arguments.base_url, enable_thinking=arguments.enable_thinking)
    else:
        raise ValueError(f"Invalid model type: {arguments.model_type}")

    evaluate(model, dataset, arguments)
