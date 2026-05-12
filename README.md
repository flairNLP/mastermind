<div align="center">

# MastermindEval

Evaluating Reasoning Capabilities of LLMs Using the Mastermind Board Game.

![Game Overview](mastermind.png)

[Installation](#🚀-installation) | [Evaluation Paradigms](#🏆-evaluation-paradigms) | [Basic Concepts](#🔑-basic-concepts) | [Running the Evaluations](#running-the-evaluations) |
[Citation](#📚-citation)

</div>

## 🚀 Installation 

To set up the environment and install dependencies, run the following:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## 🏆 Evaluation Paradigms

We provide three different evaluation paradigms:

1. **🤖 Agentic Evaluation**: The model actively plays Mastermind, interacting with the game environment.
2. **📝 Prompt-Based Evaluation**: The model is presented with pre-played game scenarios and must deduce the last possible code.
3. **🎯 Multiple-Choice Evaluation**: The model ranks different code options based on log-likelihood, aligning with pretraining objectives. You can run your evaluation using the awesome [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).
## 🔑 Basic Concepts

- **🧩 Model Class**: Defines the LLM interface that interacts with the game. We provide support for:
  - Hugging Face Model Hub
  - OpenAI
  - Anthropic
  - vLLM via its OpenAI-compatible server
- **🎲 Mastermind Game Class**: Represents a game instance with customizable parameters such as `num_colors` and `possible_colors`.
- **📊 Evaluator Class**: Manages the evaluation process by executing multiple rounds of the game and assessing model performance.

## Running the Evaluations

We provide various scripts for running different evaluation methods:

- **🤖 Agentic Evaluation**: `run_full_game.py` (Python script) and `run_full_game.sh` (Bash script)
- **📝 Prompt-Based Evaluation**: `run_instructions.py` (Python script) and `run_instructions.sh` (Bash script) - These splits are also availabe on the 🤗 [Hugging Face hub](https://huggingface.co/collections/flair/mastermindeval-67cb01daedbee142edd594ea)!
- **🎯 Multiple-Choice Evaluation**: `run_multiple_choice.sh` (Bash) – relies on `lm-eval-harness`. These splits are also availabe on the 🤗 [Hugging Face hub](https://huggingface.co/collections/flair/mastermindeval-67cb01daedbee142edd594ea)!

### Example Usage

**Hugging Face model:**

```python
from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind
from mastermind.models import HFModel
from mastermind.utils import print_summary

model = HFModel(model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
game = Mastermind(code_length=4, num_colors=6)
evaluator = Evaluator(game, model, use_cot=True, use_fewshot_example=True)
result = evaluator.run(num_games=100, save_results=True, save_path="results", compute_progress=True)
print_summary(model, game, result, num_runs=100)
```

**vLLM (OpenAI-compatible server):**

Start the server, then run via Python or CLI:

```python
from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind
from mastermind.models import VLLMModel
from mastermind.utils import print_summary

model = VLLMModel(model_name='Qwen/Qwen2.5-7B-Instruct', base_url='http://127.0.0.1:8000/v1')
game = Mastermind(code_length=4, num_colors=6)
evaluator = Evaluator(game, model, use_cot=True, use_fewshot_example=True)
result = evaluator.run(num_games=100, num_parallel=8, save_results=True, save_path="results", compute_progress=True)
print_summary(model, game, result, num_runs=100)
```

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto
python run_full_game.py --model_type vllm --model Qwen/Qwen2.5-7B-Instruct --base_url http://127.0.0.1:8000/v1 --num_parallel 8
```

> **Tip:** vLLM supports concurrent requests, so you can set `--num_parallel` to a higher value (e.g. 8–32) for much faster throughput.

---

## 📚 Citation

```
@inproceedings{
  golde2025mastermindeval,
  title={MastermindEval: A Simple But Scalable Reasoning Benchmark},
  author={Jonas Golde and Patrick Haller and Fabio Barth and Alan Akbik},
  booktitle={Workshop on Reasoning and Planning for Large Language Models},
  year={2025},
  url={https://openreview.net/forum?id=H4donosutm}
}
```

For any issues, feel free to open an issue or contribute to the repository! 🚀
