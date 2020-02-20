"""Generates LaTeX contents for table comparing DSR and GP performance."""

import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, fisher_exact

MODES = ["Nguyen", "Constant"]
THRESHOLD = 1e-10
PRECISION = 3
ROOT = "./ICML_LOGS/dsr_vs_gp/"


def main():

    methods = ["gp", "dsr"]
    expressions = {
        "Nguyen-1" : "$x^3+x^2+x$",
        "Nguyen-2" : "$x^4+x^3+x^2+x$",
        "Nguyen-3" : "$x^5+x^4+x^3+x^2+x$",
        "Nguyen-4" : "$x^6+x^5+x^4+x^3+x^2+x$",
        "Nguyen-5" : "$\\sin(x^2)\\cos(x)-1$",
        "Nguyen-6" : "$\\sin(x)+\\sin(x+x^2)$",
        "Nguyen-7" : "$\\log(x+1)+\\log(x^2+1)$",
        "Nguyen-8" : "$\\sqrt{x}$",
        "Nguyen-9" : "$\\sin(x)+\\sin(y^2)$",
        "Nguyen-10" : "$2\\sin(x)\\cos(y)$",
        "Nguyen-11" : "$x^y$",
        "Nguyen-12" : "$x^4-x^3+\\frac{1}{2}y^2-y$",
        "Constant-1" : "$3.39x^3+2.12x^2+1.78x$",
        "Constant-2" : "$\\sin(x^2)\\cos(x)-0.75$",
        "Constant-3" : "$\\sin(1.5x)\\cos(0.5y)$",
    }

    for mode in MODES:
        if mode == "Nguyen":
            names = ["Nguyen-{}".format(i+1) for i in range(12)]
        elif mode == "Constant":
            names = ["Constant-{}".format(i+1) for i in range(3)]

        # Read data
        percents = []
        means = []
        stds = []
        nrmses = []
        corrects = []
        dfs = []
        for method in methods:

            drop = ["base_r_noiseless", "r_noiseless", "base_r_test_noiseless", "r_test_noiseless"]

            path = os.path.join(ROOT, "benchmark_{}_{}.csv".format(method, mode))
            df = pd.read_csv(path)

            # Drop rows not in names
            drop = []
            for i, row in df.iterrows():
                if row["name"] not in names:
                    drop.append(i)
            df = df.drop(drop, axis=0).reindex()

            # Add NRMSE and correct column
            df["nrmse"] = np.sqrt(df["nmse"])
            if "correct_manual" in df.columns: # Manual inspection for Constant benchmarks
                df["correct"] = df["correct_manual"]
            else:
                df["correct"] = df.nrmse < THRESHOLD
            correct = df.groupby("name")["correct"]

            # To protect against outliers, floor NRMSE to 1 (equivalent to predicting the mean)
            df["nrmse"] = df["nrmse"].clip(upper=1)

            # Create groupby object
            nrmse = df.groupby("name")["nrmse"]

            # Compute statistics
            percents.append(100*correct.mean())
            means.append(nrmse.mean())
            stds.append(nrmse.std())

            nrmses.append(nrmse)
            corrects.append(correct)
            dfs.append(df)


        def format(num, threshold=THRESHOLD, precision=PRECISION):
            return "{:.{}f}".format(num, precision)


        for name in names:
            line = []

            # Name
            name_change = {n : n for n in names}
            line.append(name_change[name])

            # Expression
            line.append(expressions[name])

            for method, percent, mean, std in zip(methods, percents, means, stds):

                # Check best recovery
                bold_percent = False
                a = corrects[0].get_group(name)
                b = corrects[1].get_group(name)
                a = np.array([a.sum(), len(a) - a.sum()])
                b = np.array([b.sum(), len(b) - b.sum()])
                table = np.stack([a, b])
                p = fisher_exact(table)[1]
                if p < 0.05:
                    if method == "gp" and percent[name] > percents[1][name]:
                        bold_percent = True
                    elif method == "dsr" and percent[name] > percents[0][name]:
                        bold_percent = True

                # Check significance for NRMSE
                bold_nrmse = False
                a = nrmses[0].get_group(name)
                b = nrmses[1].get_group(name)
                p = ttest_ind(a=a, b=b, equal_var=False)[1]
                if p < 0.05:
                    if method == "gp" and mean[name] < means[1][name] and percents[1][name] < 100:
                        bold_nrmse = True
                    elif method == "dsr" and mean[name] < means[0][name] and percents[0][name] < 100:
                        bold_nrmse = True

                # Percent recovery
                s = format(percent[name], precision=0)
                if bold_percent:
                    line.append("\\textbf{{{}\\%}}".format(s))
                else:
                    line.append("{}\\%".format(s))

                # NRMSE
                s1 = format(mean[name])
                s2 = format(std[name])
                if bold_nrmse:
                    line.append("$\\mathbf{{{} \\pm {}}}$".format(s1, s2))
                else:
                    line.append("${} \\pm {}$".format(s1, s2))

            line = ' & '.join(line)
            line += ' \\\\'
            print(line)

        # Summary line
        print("\\cline{3-6}")
        line = ["", "\\multicolumn{1}{r}{Average}"]
        for i, method in enumerate(methods):

            # For "Average" line, consider each experiment an average across all expressions
            df_a = dfs[0].groupby("seed").mean() # gp
            df_b = dfs[1].groupby("seed").mean() # dsr

            if method == "gp":
                df = df_a
            elif method == "dsr":
                df = df_b

            # Check best recovery
            bold_percent = False
            a = df_a["correct"]
            b = df_b["correct"]
            p = ttest_ind(a=a, b=b, equal_var=False)[1]
            if p < 0.05:
                if method == "gp" and a.mean() > b[1].mean():
                    bold_percent = True
                elif method == "dsr" and b.mean() > a[0].mean():
                    bold_percent = True

            # Check best NRMSE
            bold_nrmse = False
            a = df_a["nrmse"]
            b = df_b["nrmse"]
            p = ttest_ind(a=a, b=b, equal_var=False)[1]
            if p < 0.05:
                if method == "gp" and a.mean() < b.mean() and b.mean() > THRESHOLD:
                    bold_nrmse = True
                elif method == "dsr" and b.mean() < a.mean() and a.mean() > THRESHOLD:
                    bold_nrmse = True

            # Percent recovery
            s = 100*df["correct"].mean()
            s = format(s, precision=1)
            if bold_percent:
                line.append("\\textbf{{{}\\%}}".format(s))
            else:
                line.append("{}\\%".format(s))

            # NRMSE
            s1 = format(df["nrmse"].mean())
            s2 = format(df["nrmse"].std())

            if bold_nrmse:
                line.append("$\\mathbf{{{} \\pm {}}}$".format(s1, s2))
            else:
                line.append("${} \\pm {}$".format(s1, s2))

        line = ' & '.join(line)
        line += ' \\\\'
        print(line)


if __name__ == "__main__":
    main()