# **Hypothesis testing via AI**

## Bitbucket repository

https://mybitbucket.llnl.gov/projects/HYPOTHESIS/repos/hypothesis_testing/browse

## Confluence page

https://myconfluence.llnl.gov/display/HYPOTHESIS/Hypothesis+testing+via+AI%3A+Generating+physically+interpretable+models+of+scientific+data+with+machine+learning

## Package overview

The package `dsr` contains the code for deep symbolic regression (DSR), including a single-point launch script (`train.py`) and parallalized batch-mode launch script (`run_benchmarks.py`).

The main dependencies are `tensorflow`, which is used for the RNN controller, `gplearn`, which is used to compare against genetic programming (GP) methods, and `sympy`, which is _only_ used for pretty printing and parsing string representations of expressions, not as a fully-functional computer algebra system.

# Installation

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Source the virtual environmnet
pip install -r requirements.txt # Install Python dependencies
```

# Getting started

## Configuring runs

`dsr` uses JSON files for configuring runs, as opposed to using a bunch of command-line arguments. This makes tracking runs easier, makes it super easy/clean to instantiate objects with many arguments, and facilitates other tasks like parameter sweeps. In general (though there are exceptions), the top-level keys of the JSON are class names, and the second-level keys are constructor keyword arguments. Note this type of configuration is simply a stylistic choice.

Top-level key "dataset" specifies the library of operators and the benchmark expression to use for DSR or GP. See docs in `dataset.py` for details.

Top-level key "training" specifies the training hyperparameters for DSR. See docs in `train.py` for details.

Top-level key "controller" specifies the RNN controller hyperparameters for DSR. See docs for in `controller.py` for details.

Top-level key "gp" specifies the hyperparameters for GP. See docs for `gplearn.genetic.SymbolicRegressor` for details.

### Configuring 

## Launching runs

After configuring a run, launching it is trivial:

```
python [SCRIPT_NAME] [PATH_TO_CONFIG]
```

## Examples

### Train DSR on one benchmark

```
python train.py config.json
```

### Train DSR on all benchmarks

```
python run_benchmarks.py config.json --method=dsr
```

### Train GP on all benchmarks

```
python run_benchmarks.py config.json --method=gp
```