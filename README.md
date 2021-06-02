# Deep symbolic regression

Deep symbolic regression (DSR) is a deep learning algorithm for symbolic regression--recovering tractable mathematical expressions from an input dataset. The package `dsr` contains the code for DSR, including a single-point, parallelized launch script (`dsr/run.py`), baseline genetic programming-based symbolic regression algorithm, and scripts to reproduce results and figures from the paper.

# Installation

From the repository root:

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Activate the virtual environmnet
pip install -r requirements.txt # Install Python dependencies
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dsr # Install DSR package
```
Note: To install in LC, use `Python 3.6.4`. You can do:
```
python3-3.6.4 -m venv venv3 # Create a Python 3.6.4 virtual environment
```

To install additional dependencies only needed for reproducing figures from the paper:

```
pip install -r requirements_plots.txt
```

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

To execute DSP (Deep Symbolic Policy) algorithm:
In config.json, set "training"->"env_params"->"set_dsp" as true

```
./run_dsp.exe # Run DSP for gym environment defined as in config.json
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

Top-level key "gp" specifies the hyperparameters for GP. See docs in `baselines/gpsr.py` for details.

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
python -m dsr.run config.json --b=Nguyen-1 --mc=2 --n_cores_task=2
```

### Train DSR on all Nguyen benchmarks using 12 cores

```
python -m dsr.run config.json --b=Nguyen... --n_cores_task=12
```

### Train 2 independent runs of GP on Nguyen-1

```
python -m dsr.run config.json --method=gp --b=Nguyen-1 --mc=2 --n_cores_task=2
```

### Train DSR on Nguyen-1 and Nguyen-4

```
python -m dsr.run config.json --b=Nguyen-1 --b=Nguyen-4
```

## Using an external dataset

- Create a directory `DATAPATH` with your data in your input deck :  `Dataset1.csv`, `Dataset2.csv`,...
- Point to that directory in your configuration file  `base.json`:
```
   "task": {
      "task_type": "regression",
      "name": null,
      "dataset": {
         "file": "benchmarks.csv",
         "name": null,
         "noise": null,
         "extra_data_dir": "DATAPATH",
         "function_set": [ 
...
...
```
- Call `dsr` by:
```
python -m dsr.run base.json --method=dsr --mc=50 --n_cores_task=24 --b=Dataset1 --output_filename run_stats_Nguyen-12_mp.1.csv
```
Note the `--b` flag matches the name of the CSV file (-`.csv` ) : `Dataset1.csv`


## Summary and evaluation of a log path

With this tool one can easily get a summary of the executed experiment that is generated from the log files.
If plots are generated they will be placed in the same log directory.
### Program integration
Printing the summary is automatically turned on as well as plotting the curves for HoF and PF if they are logged.
Can be changed in `config.json`:
```
{
   ...
   "postprocess": {
      "method": "dsr",
      "print": true,
      "print_count": 5,
      "save_plots": true
   },
   ...
}
```
### Commandline usage

```
python -m dsr.logeval path_to_log_directory --log_count 10 --show_hof --show_pf --save_plots --show_plots
```
### Jupyter notebook usage
```
from dsr.logeval import LogEval
log = LogEval(path_to_log_directory)
log.analyze_log(log_count=10, show_hof=True, show_pf=True, show_plots=True)
```

# Generating log files

You can turn on/off the generation of log files through `config.json`:
```
{
   ...
   "training": {
      "logdir": "./log",
      "hof": 100,
      "save_summary": true,
      "save_all_r": true,
      "save_positional_entropy": true,
      "save_pareto_front": true,
      "save_cache": true,
      "save_cache_r_min": 0.9
   },
   ...
}
```
`logdir`: folder where the log files are saved.

`hof`: Number of programs from the "Hall of fame" (best all times programs) to be included in the log file.

`save_summary`: Whether to store Tensorflow [summaries](https://www.tensorflow.org/api_docs/python/tf/summary)

`save_all_r`: Whether to store a `.npy` file dumping the rewards from all programs sampled throughout the training process (might result in huge files). 

`save_positional_entropy`: Whether to save the evolution of positional entropy for each iteration into a `.npy` dump file.

`save_pareto_front`: Whether to save a file listing all the programs in the pareto front.

`save_cache`: Whether to save the str, count, and r of each program in the cache.

`save_cache_r_min`: If not null, only keep Programs with r >= r_min when saving cache.

For explanations of specific fields inside those files, read the comments in `dsr/dsr/train_stats.py`