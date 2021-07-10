"""Plot barplots for ablation studies."""

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import sem

import matplotlib
import matplotlib.pyplot as plt

LOGDIR = "./log/ablations/" # Directory containing results
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

    # Abalations data
    data = {}
    data["nrmse"] = {}
    data["correct"] = {}
    dso_baseline = {"nrmse" : None, "correct" : None}

    # Run from paper directory
    for root, dirs, files in os.walk(LOGDIR):

        # Read the summary file
        df = None
        for method in ["dso", "gp"]:
            if "benchmark_{}.csv".format(method) in files:
                path = os.path.join(root, "benchmark_{}.csv".format(method))
                df = pd.read_csv(path)

        if df is None:
            continue

        # Drop rows not in names
        drop = []
        for i, row in df.iterrows():
            if row["name"] not in names:
                drop.append(i)
        df = df.drop(drop, axis=0)

        if len(df) != n_rows:
            print("Found {} rows, expected {}. Skipping {}.".format(len(df), n_rows, root))
            continue

        # Create NRMSE and correct columns
        df["nrmse"] = np.sqrt(df["nmse"]).clip(upper=1)
        df["correct"] = (df.nrmse < THRESHOLD)

        # Generate labels
        label = os.path.basename(os.path.normpath(root))
        if label.startswith("no_"):
            label = label[3:]
        label = label[:-18] # Remove timestamp
        label = label.replace('_', ' ')
        replace = {
            "improvements" : "all improvements",
            "hierarchical" : "parent/sibling",
            "risk" : "risk-seeking",
            "constraints" : "all constraints",
            "entropy" : "entropy bonus",
            "min max" : "constrain min/max",
            "inv" : "constrain inverse",
            "trig" : "constrain trig"
        }
        for k, v in replace.items():
            if label.endswith(k):
                label = label[:-len(k)] + v
        label = label.capitalize()

        for metric in ["correct", "nrmse"]:
            avg = df[metric].mean()
            err = ERROR_FUNCTION(df.groupby("seed")[metric].mean())
            data[metric][label] = (avg, err)

            # Set baseline, if appropriate
            if label == "Full":
                dso_baseline[metric] = avg

    for metric in ["correct", "nrmse"]:

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5, 2)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Order of bars; None corresponds to gap to separate groups
        first = ["Full"]
        last = ["Vanilla"]
        order = [None, "hierarchical", "risk", "entropy", "improvements", None, "inv", "min max", "trig", "constraints", None]
        order = [o if o not in replace else replace[o] for o in order]
        order = first + order + last
        indices = []
        for i, o in enumerate(order):
            if o is not None:
                indices.append(i)
        labels = [o.capitalize() for o in order if o is not None]

        # Retrieve the data
        heights = [data[metric][l][0] if l in data[metric] else 0 for l in labels]
        errs = [data[metric][l][1] if l in data[metric] else 0 for l in labels]

        # Add bars
        ax.bar(indices, heights, align='center', yerr=errs)
        width = [p.get_width() for p in ax.patches][0] # Bar width

        # Add baselines
        baseline = dso_baseline[metric]
        if baseline is not None:
            ax.axhline(baseline, color='black', linewidth=0.75, linestyle='--', zorder=0)

        # Configure axes
        labels[0] = "No ablations"
        labels[9] = "All constraints\n& improvements"
        ax.set_xticks(indices)
        skip = [False for l in labels]
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            if not skip[i]:
                label.set_rotation(40)
                label.set_ha('right')
                label.set_rotation_mode("anchor")            
        ax.set_xticklabels(labels)

        # Shift x axis ticks after rotating
        dx = 0/72.; dy = 2/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            if not skip[i]:
                label.set_transform(label.get_transform() + offset)

        if metric == "correct":
            ax.set_ylabel("Recovery")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        elif metric == "nrmse":
            ax.set_ylabel("NRMSE")

        plt.gcf().subplots_adjust(bottom=0.5)

        path = os.path.join(PREFIX, "ablations_{}.pdf".format(metric))
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()
