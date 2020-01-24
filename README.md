# **Hypothesis testing via AI**

## Bitbucket repository

https://mybitbucket.llnl.gov/projects/HYPOTHESIS/repos/hypothesis_testing/browse

## Confluence page

https://myconfluence.llnl.gov/display/HYPOTHESIS/Hypothesis+testing+via+AI%3A+Generating+physically+interpretable+models+of+scientific+data+with+machine+learning

## Package overview

The package `dsr` contains the code for deep symbolic regression (DSR), including a single-point, parallelizable launch script (`run.py`).

The main dependencies are `tensorflow`, which is used for the RNN controller, `gplearn` and `deap`, which are used to compare against genetic programming (GP) methods, and `sympy`, which is _only_ used for pretty printing and parsing string representations of expressions, not as a fully-functional computer algebra system.

# Installation

Installation is straightforward in a python3 virtual environment using pip. The only non-trivial part is the dependency `gplearn`. Since we are using a feature (the ability to disallow constants) not available on the pip release, we need to clone and install the GitHub repository rather than `pip install gplearn`.

```
git clone https://[YOUR_OUN]@mybitbucket.llnl.gov/scm/hypothesis/hypothesis_testing.git # Clone the repository
cd hypothesis_testing # Start at the top level of the repository
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Source the virtual environmnet
pip install -r requirements.txt # Install Python dependencies
git clone https://github.com/trevorstephens/gplearn.git # Clone gplearn
pip install ./gplearn # Install gplearn
python dsr/setup.py build_ext --inplace # Compile cython source
```

# Getting started

## Configuring runs

`dsr` uses a JSON file to configure training hyperparameters, as opposed to using a bunch of command-line arguments. This makes tracking runs easier, makes it super easy/clean to instantiate objects with many arguments, and facilitates other tasks like parameter sweeps. In general (though there are exceptions), the top-level keys of the JSON are class names, and the second-level keys are constructor keyword arguments. Note this type of configuration is simply a stylistic choice.

Top-level key "dataset" specifies the library of operators and the benchmark expression to use for DSR or GP. See docs in `dataset.py` for details.

Top-level key "training" specifies the training hyperparameters for DSR. See docs in `train.py` for details.

Top-level key "controller" specifies the RNN controller hyperparameters for DSR. See docs for in `controller.py` for details.

Top-level key "gp" specifies the hyperparameters for GP using `gplearn`. See docs for `gplearn.genetic.SymbolicRegressor` for details.

Top-level key "deap" specifies the hyperparameters for GP using `deap`. See docs for `dsr.baselines.deap.GP` for details.

## Launching runs

After configuring a run, launching it is trivial:

```
python -m dsr.run [PATH_TO_CONFIG] [--OTHER_FLAGS]
```

## Examples

### Show command-line help and quit

```
python -m dsr.run --help
```

### Train DSR on Korns-1

```
python -m dsr.run config.json --only=Korns-1
```

### Train DSR on all benchmarks (not recommended)

```
python dsr.run config.json
```

### Train GP (using `gplearn`) on Korns-1

```
python dsr.run config.json --method=gp --only=Korns-1
```

### Train GP (using `deap`) on Korns-1

```
python -m dsr.run config.json --method=gp --only=Korns-1
```

### Train DSR on Korns-1 and Nguyen-4

```
python dsr.run config.json --only=Korns-1 --only=Nguyen-4
```

### Train DSR on all benchamrks except those beginning with Korns (i.e. Korns-1, Korns-2, etc.)

```
python -m dsr.run config.json --exclude=Korns
```