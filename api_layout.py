# TODO: to be removed once ready
import sys; sys.path.append("src")

from mastermind import Evaluator, HFModel, Mastermind, OpenAIModel, AnthropicModel

game = Mastermind()
# model = HFModel("Qwen/Qwen2-1.5B-Instruct")
# model = OpenAIModel()
model = AnthropicModel()
evalutor = Evaluator(game, model)
result = evalutor.run(num_games=2, save_results=True)
