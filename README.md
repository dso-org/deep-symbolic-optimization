# Deep symbolic regression

Deep symbolic regression (DSR) is a deep learning algorithm for symbolic regression--recovering tractable mathematical expressions from an input dataset. The package `dsr` contains the code for DSR, including a single-point, parallelized launch script (`dsr/run.py`), baseline genetic programming-based symbolic regression algorithm, and scripts to reproduce results and figures from the paper.

This code supports the paper [Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients](https://arxiv.org/abs/1912.04871).

# Installation

Installation is straightforward in a python3 virtual environment using pip. From the repository root:

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Activate the virtual environmnet
pip install -r requirements.txt # Install Python dependencies
```

To install additional dependencies only needed for reproducing figures from the paper:

```
pip install -r requirements_plots.txt
```

To perform experiments involving the GP baseline, you will need the additional package `deap`.

# Example usage

To try out `dsr`, use the following command from the repository root:

```
python -m dsr.run ./paper/config/test.json --b=Nguyen-6
```

This should solve in around 50 training steps (~30 seconds on a laptop).

# Reproducing paper results

Results from the paper can be exactly reproduced using the following commands. First, `cd` into the `paper` directory.

To reproduce hyperparameter sweeps:

```
python setup_sweep.py # Generate config files and shell scripts to run hyperparameter sweeps
./run_sweep_dsr.sh # Run DSR hyperparameter sweep
./run_sweep_gp.sh # Run GP hyperparameter sweep
```

To reproduce DSR vs GP performance comparisons:

```
./run_Nguyen_dsr.sh # Run DSR on Nguyen benchmarks
./run_Nguyen_gp.sh # Run GP on Ngueyn benchmarks
./run_Constant_dsr.sh # Run DSR on Constant benchmarks
./run_Constant_gp.sh # Run GP on Constant benchmarks
```

To reproduce ablation studies:

```
python setup_ablations.py # Generate config files and shell scripts to run ablation experiments
./run_ablations.sh # Run DSR ablations
```

To reproduce noise and dataset size studies:

```
python setup_noise.py # Generate config files and shell scripts to run noise and dataset size experiments
./run_noise_dsr.sh # Run DSR for various noise levels and dataset sizes
./run_noise_gp.sh # Run GP for various noise levels and dataset sizes
```

Figures and tables from the paper can be reproduced using the following scripts after having run the relevant experiments:

```
python plot_table.py # Generate Table 1 contents (DSR vs GP performance comparison)
python plot_distributions.py 0 # Generate Figure 2 (risk-seeking vs standard policy gradients for Nguyen-8)
python plot_ablations.py # Generate Figure 3 (ablations barplots)
python plot_noise.py # Generate Figure 4 (noise and dataset size comparison)
python plot_training.py # Generate Figures 5 and 6 (training curves)
python plot_distributions.py 1 # Generate Figures 7 and 8 (risk-seeking vs standard policy gradients for all Nguyen benchmarks)
```

# Getting started

## Configuring runs

`dsr` uses JSON files to configure training.

Top-level key "dataset" specifies details of the benchmark expression for DSR or GP. See docs in `dataset.py` for details.

Top-level key "training" specifies the training hyperparameters for DSR. See docs in `train.py` for details.

Top-level key "controller" specifies the RNN controller hyperparameters for DSR. See docs for in `controller.py` for details.

Top-level key "gp" specifies the hyperparameters for GP. See docs for `dsr.baselines.gspr.GP` for details.

## Launching runs

After configuring a run, launching it is simple:

```
python -m dsr.run [PATH_TO_CONFIG] [--OPTIONS]
```

## Examples

### Show command-line help and quit

```
python -m dsr.run --help
```

### Train 2 indepdent runs of DSR on Nguyen-1 using 2 cores

```
python -m dsr.run config.json --b=Nguyen-1 --mc=2 --num_cores=2
```

### Train DSR on all Nguyen benchmarks using 12 cores

```
python -m dsr.run config.json --b=Nguyen --num_cores=12
```

### Train 2 independent runs of GP on Nguyen-1

```
python -m dsr.run config.json --method=gp --b=Nguyen-1 --mc=2 --num_cores=2
```

### Train DSR on Nguyen-1 and Nguyen-4

```
python -m dsr.run config.json --b=Nguyen-1 --b=Nguyen-4
```

# Release

LLNL-CODE-647188
