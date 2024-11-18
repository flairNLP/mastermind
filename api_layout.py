# TODO: to be removed once ready
from mastermind import Evaluator, HFModel, Mastermind

game = Mastermind()
model = HFModel("Qwen/Qwen2-1.5B-Instruct")
evalutor = Evaluator(game, model)
result = evalutor.run(num_games=2, save_results=True)
