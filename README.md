# Mastermind

## Setup

```
conda create -n mastermind python=3.11
conda activate mastermind
pip install -e .
```

## Run

```python
from mastermind import AnthropicModel, Evaluator, HFModel, Mastermind, OpenAIModel

game = Mastermind()
model = HFModel("Qwen/Qwen2-1.5B-Instruct", generation_args={"max_new_tokens": 1024})
# model = OpenAIModel()
# model = AnthropicModel()
evalutor = Evaluator(game, model)
result = evalutor.run(num_games=2, save_results=True)
```