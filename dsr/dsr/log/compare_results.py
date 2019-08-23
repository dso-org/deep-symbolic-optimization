import os
import json
from pkg_resources import resource_filename

import pandas as pd


def main():
    df_correct = None
    for root, dirs, files in os.walk('./Aug22'):
        
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
#        data_path = resource_filename("dsr", "data/")
#        benchmark_path = os.path.join(data_path, config["dataset"]["file"])
#        benchmark = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
#        n = len(benchmark)
        n = 12 # HACK
        if len(df) != n:
            continue

        if "base_r" not in df.columns:            
            df["base_r"] = (df["r"] + 0.001*(1 + df["traversal"].str.count(',')))

        if method == "dsr":
            r = config["training"]["reward"]
            if r.startswith("neg_"):
                df["metric"] = -df["base_r"]
            elif r.startswith("inv_"):
                df["metric"] = 1/df["base_r"] - 1
            if "nrmse" in r:
                metric = "N-R-MSE"
            elif "nmse" in r:
                metric = "N-MSE"
            elif "mse" in r:
                metric = "MSE"
        else:
            df["metric"] = df["base_r"]

        avg = df["metric"].mean()
        df["correct"] = df["metric"] < 1e-12
        n_correct = df["correct"].sum()

        correct = pd.Series(dict(zip(df.name, df.correct)), name=root)
       
        if df_correct is None:
            df_correct = pd.DataFrame(columns=df.name)
        df_correct = df_correct.append(correct)

        print("{} Avg {}: {:.6f}, Correct: {} of {} ({:.1f}%).".format(root.ljust(120), metric, avg, n_correct, n, 100*n_correct/n))

    if df_correct is not None:
        df_correct.to_csv("correct.csv")

if __name__ == "__main__":
    main()
