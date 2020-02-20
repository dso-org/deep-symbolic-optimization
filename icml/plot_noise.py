"""Plot noise and dataset size comparison results."""

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import sem

import matplotlib
import matplotlib.pyplot as plt

LOGDIR = "./ICML_LOGS/noise/" # Directory containing results
PREFIX = "plots" # Directory to save plots
ERROR_FUNCTION = sem
THRESHOLD = 1e-12
FONTSIZE = 8

matplotlib.rc('font', size=FONTSIZE)


def main():

    os.makedirs(PREFIX, exist_ok=True)

    names = ["Nguyen-{}".format(i+1) for i in range(12)]
    mc = 10
    n_rows = mc * len(names)

    multipliers = [1, 10, 100][:2]
    linestyles = ["-", "--", "."][:2]
    dfs = {}
    nrmse = {}
    correct = {}
    nrmse_sem = {}
    correct_sem = {} 
    for multiplier in multipliers:
        dfs[multiplier] = {}
        nrmse[multiplier] = {}
        correct[multiplier] = {}
        nrmse_sem[multiplier] = {}
        correct_sem[multiplier] = {} 

    for multiplier, linestyle in zip(multipliers, linestyles):

        names = ["Nguyen-{}".format(i+1) for i in range(12)]
        mc = 10
        epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        methods = ["dsr", "gp"]

        for method in methods:
            nrmse[multiplier][method] = np.zeros_like(epsilons)
            correct[multiplier][method] = np.zeros_like(epsilons)
            nrmse_sem[multiplier][method] = np.zeros_like(epsilons)
            correct_sem[multiplier][method] = np.zeros_like(epsilons)

        for i, epsilon in enumerate(epsilons):

            for m in methods:

                path = os.path.join(LOGDIR, m)
                files = os.listdir(path)
                filename = None
                for f in files:
                    params = f.split("_")[:2]
                    params = [p.strip("nd") for p in params]
                    if float(params[0]) == epsilon and int(params[1]) == multiplier:
                        filename = f
                        break
                assert filename is not None
                df = pd.read_csv("{}/{}/benchmark_{}.csv".format(path, filename, m))

                # Compute correctness, which uses base_r_test_noiseless
                if m == "dsr":
                    # ASSUMING DSR USED INV_NRMSE AS REWARD
                    df["nrmse_noiseless"] = (1/(df["base_r_test_noiseless"]) - 1).clip(upper=1)
                elif m == "gp":
                    # ASSUMING GP USED NRMSE AS FITNESS
                    df["nrmse_noiseless"] = (df["base_r_test_noiseless"]).clip(upper=1)
                df["correct"] = (df["nrmse_noiseless"] < THRESHOLD)

                # Reported NRMSE is not noiselesss
                df["nrmse"] = np.sqrt(df["nmse"]).clip(upper=1)

                nrmse[multiplier][m][i] = df["nrmse"].mean()
                nrmse_sem[multiplier][m][i] = ERROR_FUNCTION(df.groupby("seed")["nrmse"].mean())
                correct[multiplier][m][i] = df["correct"].mean()
                correct_sem[multiplier][m][i] = ERROR_FUNCTION(df.groupby("seed")["correct"].mean())

                dfs[multiplier][m] = df

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(4.5, 1.5)

    for ax, metric in zip(axes, ["correct", "nrmse"]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        for multiplier, linestyle in zip(multipliers, linestyles):

            ys = {"nrmse" : nrmse[multiplier], "correct" : correct[multiplier]}
            error = {"nrmse" : nrmse_sem[multiplier], "correct" : correct_sem[multiplier]}

            # Line plots
            logx = False
            if epsilons[0] == 0 and logx:
                epsilons[0] = epsilons[1]/10
            label = "DSR" if multiplier == multipliers[0] else None
            def unclip(e):
                for bar in e[2]:
                    bar.set_clip_on(False)
            e = ax.errorbar(epsilons, ys[metric]["dsr"], yerr=error[metric]["dsr"], label=label, color="C0", linestyle=linestyle)
            unclip(e)
            label = "GP" if multiplier == multipliers[0] else None
            e = ax.errorbar(epsilons, ys[metric]["gp"], yerr=error[metric]["gp"], label=label, color="C1", linestyle=linestyle)
            unclip(e)
            label = "{}x data".format(multiplier)
            ax.plot([],[], color="gray", linestyle=linestyle, label=label)
            if logx:
                plt.xscale("log")
            if metric == "correct" and multiplier == multipliers[-1]:
                handles, labels = ax.get_legend_handles_labels()
                order = [2, 3, 0, 1]
                legend_params = {
                    "loc" : "upper center",
                    "ncol" : 1,
                    "handlelength" : 1.75,
                    "frameon" : False,
                    "borderpad" : 0
                }
                ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], **legend_params)
            ax.set_xticks(epsilons[::2])
            ax.set_xlim((0 - 0.001, 0.1 + 0.001))
            ax.set_ylim((0, None))

            # Both
            ax.set_xlabel("Noise level")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:g}'.format(x)))
            if metric == "correct":
                ax.set_ylabel("Recovery")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            elif metric == "nrmse":
                ax.set_ylabel("NRMSE")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    path = os.path.join(PREFIX, "noise.pdf")
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()