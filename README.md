# Mastermind

## Setup

```
conda create -n mastermind python=3.11
conda activate mastermind
pip install -e .
```

## Run

Using Language Models.

```python
from mastermind.models import AnthropicModel, HFModel, OpenAIModel
from mastermind.game import Mastermind
from mastermind.evalutor import Evalutor
from mastermind.solvers import KnuthSolver

game = Mastermind()
model = HFModel("Qwen/Qwen2-1.5B-Instruct", generation_args={"max_new_tokens": 1024})
# model = OpenAIModel()
# model = AnthropicModel()
evalutor = Evaluator(game, model)
result = evalutor.run(num_games=2, save_results=True)
```

Using Solvers.
```python
from mastermind.game import Mastermind
from mastermind.solvers import KnuthSolver

game = Mastermind()
solver = KnuthSolver(game=game)
solver.solve(num_games=2, save_results=True)
```