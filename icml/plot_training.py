"""Plot training curves for DSR and GP."""

import os
import re
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset

LOGDIR = "./ICML_LOGS/dsr_vs_gp/" # Directory containing results
PREFIX = "plots" # Directory to save plots
FONTSIZE = 8
NROWS = 4
NCOLS = 4
N_EXPRESSIONS = [int(2e6)] * 12 + [int(1e6)] * 3
MC = 100 # Number of MC trials for Nguyen benchmarks
MC_CONSTANT = 10 # Number of MC trials for Constant benchmarks

matplotlib.rc('font', size=FONTSIZE)


def main():

    os.makedirs(PREFIX, exist_ok=True)

    methods = ["gp", "dsr"]
    benchmarks = ["Nguyen-{}".format(i) for i in range(1, 13)] + ["Constant-{}".format(i) for i in range(1, 4)]
    labels = {"gp" : "GP", "dsr" : "DSR"}
    colors = {"gp" : "C1", "dsr" : "C0"}

    best = {"gp" : [], "dsr" : []} # Best so far
    correct = {"gp" : [], "dsr" : []} # Fraction correct so far
    data = {"best" : best, "correct" : correct}
    data = [deepcopy(data) for _ in benchmarks]

    dfs = {}
    for i in range(12):
        dfs[i+1] = np.empty(shape=(), dtype=np.float32)

    # Constant correctness was determined by manual inspection of the symbolic expression
    # To get a threshold, check the manual correctness and retrieve the corresponding best reward
    thresholds = {"dsr" : [], "gp" : []}
    for method in methods:
        alg = {"dsr" : "dsr", "gp" : "deap"}
        path = os.path.join(LOGDIR, method, "benchmark_{}_Constant.csv".format(alg[method]))
        summary = pd.read_csv(path)
        for b in benchmarks:
            if "Nguyen" in b:
                threshold = 1e-12
                threshold = 1 / (1 + threshold)
            else:
                df = summary
                df = df[df["name"] == b]
                df = df[df["correct_manual"] == True]
                if len(df) > 0:
                    threshold = df["nmse"].values.max()
                    threshold = np.sqrt(threshold) # NMSE -> NRMSE
                    threshold = 1 / (1 + threshold)
                else:
                    threshold = 1.0
            thresholds[method].append(threshold)

    # Read summary files for Constant benchmarks, which have manual correctness
    summaries = {}
    for method in methods:
        alg = {"gp" : "deap", "dsr" : "dsr"}
        path = os.path.join(LOGDIR, method, "benchmark_{}_Constant.csv".format(alg[method]))
        summary = pd.read_csv(path)
        summaries[method] = summary


    def get_nrmse_threshold(method, b, mc, nrmse):
        """
        Check summary if this benchmark is correct. If so, the threshold is the best
        found nrmse.
        """
        if "Nguyen" in b:
            threshold = 1e-12
        else:
            summary = summaries[method]
            summary = summary[summary["name"] == b]
            summary = summary[summary["seed"] == mc]
            assert len(summary) == 1
            correct = summary["correct_manual"].values[0]
            if correct:
                threshold = nrmse[-1]
            else:
                threshold = 1e-12
        return threshold


    for i, b in enumerate(benchmarks):
        print("Reading DSR and GP data for benchmark {}...".format(b))
        for mc in range(MC):

            if "Constant" in b and mc >= MC_CONSTANT:
                continue

            # Load DSR data
            filename = "dsr_{}_{}.csv".format(b, mc)
            path = os.path.join(LOGDIR, "dsr", filename)
            df = pd.read_csv(path)        
            best = df["base_r_best"].values # Best reward
            nmse = df["nmse_best"].values # Best NMSE on test set
            nrmse = np.sqrt(nmse)
            THRESHOLD = get_nrmse_threshold("dsr", b, mc, nrmse)
            best[nrmse <= THRESHOLD] = 1.0
            correct = nrmse <= THRESHOLD
            data[i]["best"]["dsr"].append(best)
            data[i]["correct"]["dsr"].append(correct)

            # Load corresponding GP data
            filename = filename.replace("dsr", "deap")
            path = os.path.join(LOGDIR, "gp", filename)
            df = pd.read_csv(path)
            nmse = df["fit_best"].values[1:] # Skip initial row (before first GP update)
            nrmse = np.sqrt(nmse)
            best = 1 / (1 + nrmse) # NRMSE -> reward
            THRESHOLD = get_nrmse_threshold("gp", b, mc, nrmse)
            best[nrmse <= THRESHOLD] = 1.0
            correct = nrmse <= THRESHOLD
            data[i]["best"]["gp"].append(best)
            data[i]["correct"]["gp"].append(correct)    

    # Pad best/correct plots that ended due to early stopping
    # Early stopping is based on NRMSE on noiseless data
    early_stopping = True
    if early_stopping:
        for method in methods:
            for i in range(len(benchmarks)):
                N = 4000 if method == "dsr" else 2000 # Number of iterations/generations
                if i >= 12:
                    N = N // 2
                for j in range(len(data[i]["best"][method])):
                    x = data[i]["best"][method][j]
                    if len(x) < N:
                        print("{}_{}-{} stopped at step {}".format(method, benchmarks[i], j, len(x)))
                        new_x = np.ones(N, dtype=np.float32)
                        new_x[:len(x)] = x
                        data[i]["best"][method][j] = new_x

                        new_correct = np.ones(N, dtype=bool)
                        new_correct[:len(x)] = False
                        data[i]["correct"][method][j] = new_correct


    # REWARD PLOTS
    metric = "best"
    fig, axes = plt.subplots(NROWS, NCOLS)
    fig.set_size_inches(8, 6)
    log = True
    for i, b in enumerate(benchmarks):

        # Configure axis
        row = i // NCOLS
        col = i % NCOLS
        ax = axes[row, col]
        ax.set_title(b)
        if log:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "{:g}M".format(int(x) / 1e6)))
        if row == NROWS - 1 or (col == NCOLS - 1 and row == NROWS - 2):        
            ax.set_xlabel("Expressions evaluated")
        else:
            ax.xaxis.set_ticklabels([])
        if col == 0:
            ax.set_ylabel("Reward")
        else:
            ax.yaxis.set_ticklabels([])
        ax.set_xlim((1, max(N_EXPRESSIONS)))
        ymin = 0.5
        def ymax(ymin):
            return 1 + (1.0 - ymin)*0.08
        ax.set_ylim((ymin, ymax(ymin)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(1.0, linestyle="dashed", color="black", linewidth=0.8, zorder=-np.inf)

        # Plot data
        axinset = None
        for method in methods:
            ys = data[i][metric][method]
            ys = np.vstack(ys)
            y_mean = ys.mean(axis=0) # Mean across MC trials
            y_std = ys.std(axis=0) # Std across MC trials
            linestyle = "solid"
            color = colors[method]
            N = len(y_mean) # Number of iterations/generations
            xs = np.linspace(start=1, stop=N_EXPRESSIONS[i], num=N, endpoint=True, dtype=np.int32)
            ax.fill_between(xs, y_mean - y_std, y_mean + y_std, alpha=0.5, facecolor=color) # Stdev band
            ax.plot(xs, y_mean, label=labels[method], color=color, linestyle=linestyle) # Mean curve

            # Zoomed inset axis
            if i == 12:
                ymin = 0.999
                xmin = 1e4
                xmax = 1e5
                bbox1 = 0.143      
            elif i == 13:
                ymin = 0.99995
                xmin = 1e5
                xmax = 2e5
                bbox1 = 0.378
            elif i == 14:
                ymin = 0.998
                xmin = 6e5
                xmax = 1e6
                bbox1 = 0.615
            else:
                bbox1 = None
            if bbox1 is not None:
                if axinset is None:
                    axinset = inset_axes(ax, 1, 0.5, loc=2, bbox_to_anchor=(bbox1, 0.22), bbox_transform=ax.figure.transFigure)
                    axinset.set_xlim((xmin, xmax))
                    axinset.set_ylim((ymin, ymax(ymin)))
                    axinset.axhline(1.0, linestyle="dashed", color="black", linewidth=0.8, zorder=-np.inf)
                    if log:
                        axinset.set_xscale("log")
                    mark_inset(ax, axinset, loc1=1, loc2=2, fc="none", ec="0.5")                
                # axinset.fill_between(range(N), mean - std, mean + std, alpha=0.5, facecolor=color) # Stdev band
                axinset.plot(xs, y_mean, color=color, linestyle=linestyle) # Mean curve
                
        if axinset is not None:
            axinset.tick_params(axis='both', which='major', labelsize=6)
            axinset.set_yticks([ymin, 1])
            axinset.set_xticklabels([], minor=True) # For some reason, minor tick labels are turned on
            axinset.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))

    # Fake plots for legend
    def legend_only(ax):
        ax.axis("off")
        for method in methods:
            color = colors[method]
            ax.plot([], label=labels[method])
        ax.legend(loc="center", frameon=False)

    # Configure plot
    legend_only(axes[-1][-1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    path = os.path.join(PREFIX, "training_r.pdf")
    plt.savefig(path)

    # RECOVERY PLOTS
    metric = "correct"
    fig, axes = plt.subplots(NROWS, NCOLS)
    fig.set_size_inches(8, 6)
    log = False
    for i in range(len(benchmarks)):

        # Configure axis
        row = i // NCOLS
        col = i % NCOLS
        ax = axes[row, col]
        ax.set_title(benchmarks[i])
        if log:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "{:g}M".format(int(x) / 1e6)))
        if row == NROWS - 1 or (col == NCOLS - 1 and row == NROWS - 2):        
            ax.set_xlabel("Expressions evaluated")
        else:
            ax.xaxis.set_ticklabels([])
        if col == 0:
            ax.set_ylabel("Recovery rate")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        else:
            ax.yaxis.set_ticklabels([])

        ax.set_xlim((1, max(N_EXPRESSIONS)))
        ax.set_ylim((0, 1.01))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(1.0, linestyle="dashed", color="black", linewidth=0.8, zorder=-np.inf)

        # Plot data
        for method in methods:
            ys = data[i][metric][method]
            ys = np.vstack(ys)
            ys = ys.mean(axis=0)
            N = len(ys) # Number of iterations/generations
            xs = np.linspace(start=1, stop=N_EXPRESSIONS[i], num=N, endpoint=True, dtype=np.int32)        
            ax.plot(xs, ys, label=method.upper(), color=colors[method]) # Mean curve 

    # Configure plot
    legend_only(axes[-1][-1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    path = os.path.join(PREFIX, "training_recovery.pdf")
    plt.savefig(path)


if __name__ == "__main__":
    main()