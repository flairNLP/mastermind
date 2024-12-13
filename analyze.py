import json
import numpy as np
import pandas as pd
import glob


def extract_instruction_results():
    dirs = glob.glob("results/instructions/*")
    table = pd.DataFrame(columns=["Model", "Game", "Valid", "Solved"])
    for dir in dirs:
        with open(f"{dir}/results.json") as f:
            data = json.load(f)

        with open(f"{dir}/info.json") as f:
            info = json.load(f)

        game_info = info["dataset"].split("_")[1]
        game_info = f"c={game_info[0]}, n={game_info[1]}"
        model_info = info["model"]

        scores = {"valid": [], "solved": [], "num_guesses": [], "progress_history": []}
        for result in data:
            scores["valid"].append(1 if result["valid"] else 0)
            if result["valid"]:
                scores["solved"].append(1 if result["correct"] else 0)

        scores["valid"] = f"Valid: {np.mean(scores['valid']) * 100:.2f}"
        scores["solved"] = f"Solved: {np.mean(scores['solved']) * 100:.2f}"

        table = pd.concat(
            [table, pd.DataFrame([model_info, game_info, scores["valid"], scores["solved"]], index=table.columns).T]
        )
    table = table.pivot(index="Model", columns="Game", values=["Solved", "Valid"]).fillna("-")
    table = table.replace({"Solved:": "", "Valid:": ""}, regex=True)
    table.index = table.index.str.replace("HF Model: ", "", regex=True)
    table.index = table.index.str.replace("OpenAI Model: ", "", regex=True)
    table.reindex(
        [
            "HF Model: Qwen/Qwen2.5-3B-Instruct",
            "HF Model: meta-llama/Llama-3.2-3B-Instruct",
            "HF Model: microsoft/Phi-3.5-mini-instruct",
            "HF Model: Qwen/Qwen2.5-7B-Instruct",
            "HF Model: meta-llama/Llama-3.1-8B-Instruct",
            "HF Model: allenai/Llama-3.1-Tulu-3-8B",
            "OpenAI Model: gpt-4o",
        ]
    )

    print(table.to_latex())


def extract_eval_harness_results():
    dirs = glob.glob("results/eval_harness/*")
    table = pd.DataFrame(columns=["Model", "Game", "Solved"])

    for dir in dirs:
        with open(dir) as f:
            data = json.load(f)

        _, game_info_org, model_info = dir.split("/")[-1].split("_")
        game_info_str = f"c={game_info_org[0]}, n={game_info_org[1]}"
        model_info = model_info.replace(".json", "")

        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        model_info,
                        game_info_str,
                        round(data["results"][f"mastermind_{game_info_org}"]["acc,none"] * 100, 2),
                    ],
                    index=table.columns,
                ).T,
            ]
        )

    table = table.pivot(index="Model", columns="Game", values="Solved").fillna("-")
    table = table.reindex(
        [
            "llama-3.2-3b",
            "qwen-2.5-3b",
            "phi-3.5",
            "llama-3.1-8b",
            "qwen-2.5-7b",
            "tulu3-8b",
        ]
    )
    print(table.to_latex())


if __name__ == "__main__":
    # extract_instruction_results()
    extract_eval_harness_results()
    print()
