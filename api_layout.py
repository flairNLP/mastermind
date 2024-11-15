# to be removed once ready
from mastermind import Evaluator, HFModel, Mastermind

game = Mastermind()
model = HFModel("gpt2")
evalutor = Evaluator(game, model)
evalutor.run()
