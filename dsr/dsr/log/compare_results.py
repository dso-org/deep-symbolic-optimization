import os
import json

import pandas as pd


def main():
    df_correct = None
    for root, dirs, files in os.walk('.'):
        
        if "config.json" not in files:
            continue
        if "benchmark_dsr.csv" in files:
            method = "dsr"
        elif "benchmark_gp.csv" in files:
            method = "gp"
        else:
            continue

        df = pd.read_csv(os.path.join(root, "benchmark_{}.csv".format(method)))
        with open(os.path.join(root, "config.json"), encoding='utf-8') as f:
            config = json.load(f)
        benchmark = os.path.join("..", config["dataset"]["file"])
        benchmark = pd.read_csv(benchmark, encoding="ISO-8859-1")
        n = len(benchmark)
        if len(df) != n:
            continue

        if "base_r" not in df.columns:            
            df["base_r"] = (df["r"] + 0.001*(1 + df["traversal"].str.count(',')))

        if method == "dsr":
            if config["training"]["reward"] == "neg_mse":
                df["mse"] = -df["base_r"]
            elif config["training"]["reward"] == "inv_mse":
                df["mse"] = 1/df["base_r"] - 1
        else:
            df["mse"] = df["base_r"]

        avg = df["mse"].mean()
        df["correct"] = df["mse"] < 1e-12
        n_correct = df["correct"].sum()

        correct = pd.Series(dict(zip(df.name, df.correct)), name=root)
       
        if df_correct is None:
            df_correct = pd.DataFrame(columns=df.name)
        df_correct = df_correct.append(correct)

        print("{} Avg MSE: {:.6f}, Correct: {} of {} ({:.1f}%).".format(root.ljust(100), avg, n_correct, n, 100*n_correct/n))

    if df_correct is not None:
        df_correct.to_csv("correct.csv")

if __name__ == "__main__":
    main()
