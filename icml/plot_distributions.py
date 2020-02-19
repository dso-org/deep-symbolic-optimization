"""Plot distributions and expectations of rewards for risk-seeking vs standard policy gradient."""

import os
import sys

import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, sem
import numpy as np
import pandas as pd
from progress.bar import Bar

LOGDIR = "./ICML_LOGS/dist/" # Directory containing results
PREFIX = "plots" # Directory to save plots
RESOLUTION = 10 # Number of points in KDE estimate
LINEWIDTH = 0.75 # Linewidth used for KDEs
FONTSIZE = 8

# Generate supplement vs body versions of figures
if len(sys.argv) > 1:
    SUPPLEMENT = bool(sys.argv[1])
else:
    SUPPLEMENT = False

matplotlib.rc('font', size=FONTSIZE)


def main():

    os.makedirs(PREFIX, exist_ok=True)

    # TBD: Read parameters from config
    epsilon = 0.1
    batch_size = 500
    sub_batch_size = int(epsilon * batch_size)
    mc = 2
    n_samples = int(2e6)
    n_epochs = n_samples // batch_size
    if SUPPLEMENT:
        benchmarks = [i + 1 for i in range(12)]
    else:
        benchmarks = [8]

    path = LOGDIR
    experiments = {}
    names = ["risk-seeking", "standard"]
    for directory in os.listdir(path):
        for name in names:
            if name in directory:
                experiments[name] = {"logdir" : os.path.join(path, directory)}

    # Generate shared training curves figure for supplement
    if SUPPLEMENT:
        NROWS = 4
        NCOLS = len(benchmarks) // NROWS
        fig_supplement, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=True)
        fig_supplement.set_size_inches(8, 8)

    for b in benchmarks:
        print("Starting benchmark {}".format(b))
        for name, exp in experiments.items():
            
            logdir = exp["logdir"]
        
            # Distributions
            dist_full = [] # All values
            dist_eps = [] # Top (1 - epsilon)-quantiles

            # Statistics
            mean_full = [] # Mean of full batch
            mean_eps = [] # Mean of top epsilon fraction of batch
            max_batch = [] # Max of batch
            best = [] # Best so far
            stdev_full = [] # Stdev
            stdev_eps = [] # Conditional stdev
            quantile = [] # (1 - epsilon)-quantile

            # Read data            
            bar = Bar("Reading {} data".format(name))
            for i in range(mc):
                filename = os.path.join(logdir, "dsr_Nguyen-{}_{}_all_r.npy".format(b, i))
                data = np.load(filename)
                data = data[:n_epochs] # In case experiment ran longer than plots
                data = np.sort(data, axis=1)
                sub_data = data[:, -sub_batch_size:]

                # Retrieve distributions
                dist_full.append(data)
                dist_eps.append(sub_data)

                # Retrieve statistics                
                mean_full.append(np.mean(data, axis=1))
                mean_eps.append(np.mean(sub_data, axis=1))
                max_batch.append(np.max(data, axis=1))
                best.append(pd.Series(np.max(data, axis=1)).cummax().values)                
                quantile.append(np.min(sub_data, axis=1))

                bar.next()
            bar.finish()
            
            dist_full = np.hstack(dist_full)
            dist_eps = np.hstack(dist_eps)
            mean_full = np.mean(np.stack(mean_full), axis=0)
            mean_eps = np.mean(np.stack(mean_eps), axis=0)
            max_batch = np.mean(np.stack(max_batch), axis=0)
            best = np.mean(np.stack(best), axis=0)
            quantile = np.mean(np.stack(quantile), axis=0)

            # Add to experiments dict
            exp["dist_full"] = dist_full
            exp["dist_eps"] = dist_eps
            exp["mean_full"] = mean_full
            exp["mean_eps"] = mean_eps
            exp["max_batch"] = max_batch
            exp["best"] = best

        # Set up plots
        colorbar_width = 2 # Percent
        plot_width = (100 - colorbar_width) / 2
        width_ratios = [plot_width]*2 + [colorbar_width]

        fig = plt.figure()
        gs_main = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[99, 1], wspace=0.025)
        gs_plots = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0], wspace=0.13, hspace=0.15)
        gs_colorbar = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1], hspace=0.15)

        # fig, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': width_ratios, 'wspace' : 0.20, 'hspace' : 0.15})

        fig.set_size_inches(6.75, 2.5)

        # Color map for different experiments
        cmaps = {
            "risk-seeking" : "Greys",
            "standard" : "Blues",
            "pqt" : "Greens"
        }

        colors = {
            "risk-seeking" : "black",
            "standard" : "blue",
            "pqt" : "green"
        }

        method_names = {
            "risk-seeking" : "Risk-seeking PG",
            "standard" : "Standard PG",
            "pqt" : "Priority queue training"
        }
        
        # Plot distributions
        for row, name in enumerate(names):
            exp = experiments[name]
            cmap = matplotlib.cm.get_cmap(cmaps[name])
            color = colors[name]

            # Edit colormap to start above its starting value
            start = 0.1
            cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(start, 1.0, 256)))

            for col, mode in enumerate(["full", "eps"]):
                dist = exp["dist_{}".format(mode)]
                mean = exp["mean_{}".format(mode)]

                # Select axis
                ax = fig.add_subplot(gs_plots[row, col])
                ax.set_xlim((0.0, 1.0))

                # Plot distributions
                ylim_max = 0.0
                n_histograms = 100
                by = n_epochs // n_histograms
                epochs_to_plot = list(range(0, len(dist), by))
                bar = Bar("Computing kernel for method={}, distribution={}".format(name, mode))
                for epoch in epochs_to_plot:
                    data = dist[epoch] + 1e-6 * np.random.random(size=(dist.shape[1],))
                    kernel = gaussian_kde(data, bw_method=0.25)
                    x = np.linspace(0, 1, RESOLUTION)
                    y = kernel(x)
                    
                    # Plot distribution
                    color = cmap(epoch/n_epochs)
                    ax.plot(x, y, color=color, linewidth=LINEWIDTH)

                    # Get max ylim
                    ylim = ax.get_ylim()
                    ylim_max = max(ylim[1], ylim_max)
                    bar.next()
                bar.finish()

                # Adjust ylim
                ylim_max = min(ylim_max, 25.0)            
                ax.set_ylim((0.0, ylim_max))

                # Add text "legend"
                method_name = method_names[name]
                x_shift = 0.04
                if col == 0:
                    ax.set_ylabel("Density")
                    batch_name = "Full batch"
                    x_pos = 1 - x_shift
                    halign = "right"
                else:
                    batch_name = r"Top $\epsilon$ batch"
                    x_pos = 1 - x_shift
                    halign = "right"
                if row == 0:
                    ax.xaxis.set_ticklabels([])
                elif row == len(names) - 1:
                    ax.set_xlabel("Reward")
                text = method_name + ",\n" + batch_name
                ax.text(x_pos, 0.9*ylim_max, method_name + ",", verticalalignment='top', horizontalalignment=halign, color=cmap(1.0))
                ax.text(x_pos, 0.75*ylim_max, batch_name, verticalalignment='top', horizontalalignment=halign)

                # Add final expectation marker
                x_val = dist[-1].mean()
                y_val = 0.0 - ylim_max*0.03
                ax.scatter(x_val, y_val, color=cmap(1.0), marker="^", clip_on=False, zorder=10, s=15)

                # Add A-D label
                if SUPPLEMENT:
                    if row == 0 and col == 0:
                        benchmark_name = "Nguyen-{}".format(b)
                        ax.text(0.03, 0.82, benchmark_name, transform=ax.transAxes, fontsize=10, weight="bold")
                else:
                    letter = ["A", "B", "C", "D"][row*2 + col]
                    ax.text(0.03, 0.82, letter, transform=ax.transAxes, fontsize=10, weight="bold")

            # Add colorbar
            if row < len(names):
                # c_ax = axes[row, -1] # First column is colorbar
                c_ax = fig.add_subplot(gs_colorbar[row])
                matplotlib.colorbar.ColorbarBase(c_ax, cmap=cmap, orientation="vertical")            
                # c_ax.set_ylabel("Iteration")
                labels = [None]*len(c_ax.get_yticklabels())
                labels[0] = 1
                labels[-1] = "{}k".format(n_epochs // 1000)
                c_ax.set_yticklabels(labels)

        # Align y-axis labels
        fig.align_ylabels()

        # Save distributions figure
        
        if SUPPLEMENT:
            filename = "distributions_{}_supplement.pdf".format(b)
        else:
            filename = "distributions_{}.pdf".format(b)
        filename = os.path.join(PREFIX, filename)
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Setup figure
        if SUPPLEMENT:
            ax = axes.reshape(-1)[b-1]
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.set_size_inches(2.5, 2.5)
        
        # Plot means and bests
        for name, exp in experiments.items():
            color = colors[name]

            by = 50 # Need to downsampled dotted/dashed lines to prevent artifacts
            xs = np.arange(1, n_epochs + 1)
            
            # Plot full batch mean
            mean = exp["mean_full"]
            plot_full, = ax.plot(xs[::by], mean[::by], clip_on=False, color=color, linestyle="dotted")

            # Plot epsilon batch mean
            mean = exp["mean_eps"]
            plot_eps, = ax.plot(xs[::by], mean[::by], clip_on=False, color=color, linestyle="dashed")

            # # Plot batch max
            # mean = exp["max_batch"]
            # plot_max, = ax.plot(xs[::by], mean[::by], clip_on=False, color=color, linestyle="dashdot")

            # Plot best  
            by = 1 # Don't need to downsample here, since line is solid
            best = exp["best"]
            plot_best, = ax.plot(xs[::by], best[::by], clip_on=False, color=color, linestyle="solid")

        # Add legend for std, risk
        legend_params = {
            "loc" : "lower left",
            "frameon" : False,
            "handletextpad" : 0,
            "borderpad" : 0
        }
        for name in names:
            color = colors[name]
            label = method_names[name]
            ax.scatter([-1], [-1], marker="s", color=color, label=label)
        legend1 = ax.legend(**legend_params)

        # Add legend for full, eps, best
        legend_params = {
            "loc" : "lower right",
            "frameon" : False,
            "borderpad" : 0,
            "handlelength" : 1.75
        }
        # Fake plots with gray color
        plot_full, = ax.plot([-1], color="gray", linestyle="dotted")
        plot_eps, = ax.plot([-1], color="gray", linestyle="dashed")
        plot_best, = ax.plot([-1], color="gray", linestyle="solid")
        plots = [plot_full, plot_eps, plot_best]
        labels = ["Full", r"Top $\epsilon$", "Best"]
        ax.legend(plots, labels, **legend_params) # Removes previous legend
        ax.add_artist(legend1)

        # Format axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(1, n_epochs)
        ax.set_ylim(0, 1)
        if SUPPLEMENT:
            ax.set_title("Nguyen-{}".format(b))
            if ax in axes[:, 0]: # First column
                ax.set_ylabel("Reward")
            if ax in axes[-1, :]: # Last row
                ax.set_xlabel("Iterations")
        else:
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Reward")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x) // 1000) + 'k'))

        # Add E label
        if not SUPPLEMENT:
            ax.text(-0.225, 0.965, "E", transform=ax.transAxes, fontsize=10, weight="bold")

        if not SUPPLEMENT:
            filename = "curves_{}.pdf".format(b)
            filename = os.path.join(PREFIX, filename)
            fig.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    if SUPPLEMENT:
        filename = "curves_supplement.pdf"
        filename = os.path.join(PREFIX, filename)
        fig_supplement.subplots_adjust(wspace=0.15, hspace=0.25)
        fig_supplement.savefig(filename, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
