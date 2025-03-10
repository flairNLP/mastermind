import json
from argparse import ArgumentParser

from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind
from mastermind.models import AnthropicModel, HFModel, OpenAIModel
from mastermind.solvers import KnuthSolver
from mastermind.utils import print_summary

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--model_type", type=str, default="hf", help="Model type.")
    parser.add_argument("--generation_args", type=str, default={}, help="Generation kwargs.")
    parser.add_argument("--use_cot", action="store_true", help="Use COT.")
    parser.add_argument("--use_full_example", action="store_true", help="Use full example.")
    parser.add_argument("--code_length", type=int, default=4, help="Code length of the game.")
    parser.add_argument("--num_colors", type=int, default=6, help="Number of colors in the game.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--save_results", action="store_true", help="Save results.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results.")
    args = parser.parse_args()

    if args.generation_args:
        generation_args = json.loads(args.generation_args)  # Convert JSON string to a dictionary
    else:
        generation_args = None

    game = Mastermind(code_length=args.code_length, num_colors=args.num_colors)

    if args.model_type == "hf":
        model = HFModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "openai":
        model = OpenAIModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "anthropic":
        model = AnthropicModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "knuth":
        model = KnuthSolver(game)

    evaluator = Evaluator(game, model, use_cot=args.use_cot, use_fewshot_example=args.use_full_example)
    result = evaluator.run(
        num_games=args.num_runs, save_results=args.save_results, save_path=args.save_path, compute_progress=True
    )
    print_summary(model, game, result, args.num_runs)
