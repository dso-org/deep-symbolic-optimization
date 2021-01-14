# Deep symbolic regression

Deep symbolic regression (DSR) is a deep learning algorithm for symbolic regression--the task of recovering tractable mathematical expressions from an input dataset. The package `dsr` contains the code for DSR, including a single-point, parallelized launch script (`dsr/run.py`), baseline genetic programming-based symbolic regression algorithm, and an sklearn-like interface for use with your own data.

This code supports the ICLR 2021 paper [Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients](https://openreview.net/forum?id=m5Qsh0kBQG).

# Installation

Installation is straightforward in a Python 3 virtual environment using Pip. From the repository root:

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Activate the virtual environmnet
pip install -r requirements.txt # Install Python dependencies
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dsr # Install DSR package
```

To perform experiments involving the GP baseline, you will need the additional package `deap`.

# Example usage

To try out DSR, use the following command from the repository root:

```
python -m dsr.run ./dsr/dsr/config.json --b=Nguyen-6
```

This should solve in around 50 training steps (~30 seconds on a laptop).

# Getting started

## Configuring runs

DSR uses JSON files to configure training.

Top-level key "task" specifies details of the benchmark expression for DSR or GP. See docs in `regression.py` for details.

Top-level key "training" specifies the training hyperparameters for DSR. See docs in `train.py` for details.

Top-level key "controller" specifies the RNN controller hyperparameters for DSR. See docs for in `controller.py` for details.

Top-level key "gp" specifies the hyperparameters for GP if using the GP baseline. See docs for `dsr.baselines.gspr.GP` for details.

## Launching runs

After configuring a run, launching it is simple:

```
python -m dsr.run [PATH_TO_CONFIG] [--OPTIONS]
```

## Sklearn interface

DSR also provides an [sklearn-like regressor interface](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html). Example usage:

```
from dsr import DeepSymbolicRegressor
import numpy as np

# Generate some data
np.random.seed(0)
X = np.random.random((10, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2

# Create the model
model = DeepSymbolicRegressor("config.json")

# Fit the model
model.fit(X, y) # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions
model.predict(2 * X)
```

## Using an external dataset

To use your own dataset, simply provide the path to the `"dataset"` key in the config, and give your task an arbitary name.

```
"task": {
    "task_type": "regression",
    "name": "my_task",
    "dataset": "./path/to/my_dataset.csv",
    ...
}
```

Then run DSR:

```
python -m dsr.run path/to/config.json
```

Note the `--b` flag matches the name of the CSV file (-`.csv` ).

## Command-line examples

Show command-line help and quit

```
python -m dsr.run --help
```

Train 2 indepdent runs of DSR on the Nguyen-1 benchmark using 2 cores

```
python -m dsr.run config.json --b=Nguyen-1 --mc=2 --num_cores=2
```

Train DSR on all 12 Nguyen benchmarks using 12 cores

```
python -m dsr.run config.json --b=Nguyen --num_cores=12
```

Train 2 independent runs of GP on Nguyen-1

```
python -m dsr.run config.json --method=gp --b=Nguyen-1 --mc=2 --num_cores=2
```

Train DSR on Nguyen-1 and Nguyen-4

```
python -m dsr.run config.json --b=Nguyen-1 --b=Nguyen-4
```

# Release

LLNL-CODE-647188
