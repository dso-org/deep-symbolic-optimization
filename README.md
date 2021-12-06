# Deep symbolic optimization

<p align="center">
<img src="banner.png" width=750/>
</p>

Deep symbolic optimization (DSO) is a deep learning framework for symbolic optimization tasks. The package `dso` includes the core symbolic optimization algorithms, as well as support for two particular symbolic optimization tasks: (1) _symbolic regression_ (recovering tractable mathematical expressions from an input dataset) and (2) discovering _symbolic policies_ for reinforcement learning environments. In the code, these tasks are referred to as `regression` and `control`, respectively. We also include a simple interface for defining new tasks.

This repository contains code supporting the following publications:
1. Petersen et al. 2021 **Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients.** *ICLR 2021.* [Oral](https://iclr.cc/virtual/2021/poster/2578) [Paper](https://openreview.net/forum?id=m5Qsh0kBQG)
2. Landajuela et al. 2021 **Discovering symbolic policies with deep reinforcement learning.** *ICML 2021.* [Paper](http://proceedings.mlr.press/v139/landajuela21a/landajuela21a.pdf)
3. Mundhenk et al. 2021 **Symbolic Regression via Neural-Guided Genetic Programming Population Seeding.** *NeurIPS 2021* [Paper](https://arxiv.org/abs/2111.00053)
4. Landajuela et al. 2021 **Improving exploration in policy gradient search: Application to symbolic optimization.** *Math-AI @ ICLR 2021.* [Paper](https://mathai-iclr.github.io/papers/papers/MATHAI_16_paper.pdf)
5. Petersen et al. 2021 **Incorporating domain knowledge into neural-guided search via *in situ* priors and constraints** *AutoML @ ICML 2021.* [Paper]()
6. Kim et al. 2021 **Distilling Wikipedia mathematical knowledge into neural network models.** *Math-AI @ ICLR 2021.* [Paper](https://mathai-iclr.github.io/papers/papers/MATHAI_15_paper.pdf)
7. Kim et al. 2020 **An interactive visualization platform for deep symbolic regression.** *IJCAI 2020.* [Paper](https://www.ijcai.org/Proceedings/2020/0763.pdf)

# Installation

### Installation - core package

The core package has been tested on Python3.6+ on Unix and OSX. To install the core package (and the default `regression` task), we highly recommend first creating a Python 3 virtual environment, e.g.

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Activate the virtual environment
```
Then, from the repository root:
```
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dso # Install DSO package and core dependencies
```

The `regression` task is installed by default. It doesn't require any of the installation options below.

### Installation - `control` task
There are a few additional dependencies to run the `control` task. Install them using:
```
pip install -e ./dso[control]
```

### Installation - all tasks
To install all dependencies for all tasks, use the `all` option:

```
pip install -e ./dso[all]
```

# Getting started

DSO relies on configuring runs via a JSON file, then launching them via a simple command-line or a few lines of Python.

### Method 1: Running DSO via command-line interface

After creating your config file, simply run:
```
python -m dso.run path/to/config.json
```
After training, results are saved to a timestamped directory in the path given in the `"logdir"` parameter (default `./log`).

### Method 2: Running DSO via Python interface

The Python interface lets users instantiate and customize DSO models via Python scripts, an interactive Python shell, or an iPython notebook. The core DSO model is `dso.core.DeepSymbolicOptimizer`. After creating your config file, you can use:
```
from dso import DeepSymbolicOptimizer

# Create and train the model
model = DeepSymbolicOptimizer("path/to/config.json")
model.train()
```
After training, results are saved to a timestamped directory in the path given in `config["training"]["logdir"]` (default `./log`).

### Configuring runs

A single JSON file is used to configure each run. This file specifies the symbolic optimization task and all hyperparameters. 

Each configuration JSON file has a number of top-level keys that control various parts of the DSO framework. The important top-level keys are:
* `"experiment"` configures the experiment, namely the log directory and random number seed.
* `"task"` configures the task, e.g. the dataset for symbolic regression, or the Gym environment for the `control` task. See below for task-specific configuration.
* `"training"` configures training hyperparameters like `"n_samples"` (the total number of samples to generate) and `"epsilon"` (the risk factor used by the risk-seeking policy gradient).
* `"controller"` configures RNN hyperparameters like `"learning_rate"` and `"num_layers"`.
* `"prior"` configures the priors and constraints on the search space.

Any parameters not included in your config file assume default values found in `config/config_common.json`, `config/config_regression.json` (for `regression` runs), and `config/config_control.json` (for `control` runs).

##### Configuring runs for symbolic regression

Here are simple example contents of a JSON file for the `regression` task:

```
{
  "task" : {
    "task_type" : "regression",
    "dataset" : "path/to/my_dataset.csv",
    "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]
  }
}
```
This configures DSO to learn symbolic expressions to fit your custom dataset, using the tokens specified in `function_set` (see `dso/functions.py` for a list of supported tokens).

If you want to include optimized floating-point constants in the search space, simply add `"const"` to the `function_set` list. Note that constant optimization uses an inner-optimization loop, which leads to much longer runtimes (~hours instead of ~minutes).

You can test symbolic regression out of the box with a default configuration, after running setup, with a command such as:

```
python -m dso.run dso/config/config_regression.json --b Nguyen-7
```

This will run DSO on the regression task with benchmark Nguyen-7.

##### Configuring runs for learning symbolic control policies

Here's a simple example for the `control` task:
```
{
  "task" : {
    "task_type" : "control",
    "env" : "MountainCarContinuous-v0",
    "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 1.0, 5.0, 10.0]
    }
  }
}
```
This configures DSO to learn a symbolic policy for MountainCarContinuous-v0, using the tokens specified in `function_set` (see `dso/functions.py` for a list of supported tokens).

For environments with multi-dimensional action spaces, DSO requires a pre-trained "anchor" policy. DSO is run once per action dimension, and the `"action_spec"` parameter is updated each run. For an environment with `N` action dimesions, `"action_spec"` is a list of length `N`. A single element should be `null`, meaning that is the symbolic action to be learned. Any number of elements can be `"anchor"`, meaning the anchor policy will determine those actions. Any number of elements can be expression traversals (e.g. `["add", "x1", "x2"]`), meaning that fixed symbolic policy will determine those actions.

Here's an example workflow for HopperBulletEnv-v0, which has three action dimensions. First, learn a symbolic policy for the first action by running DSO with a config like:
```
{
  "task" : {
    "task_type" : "control",
    "name" : "HopperBulletEnv-v0",
    "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 1.0, 5.0, 10.0],
    "action_spec" : [null, "anchor", "anchor"],
    "anchor" : "path/to/anchor.pkl"
    }
  }
}
```
where `"path/to/anchor.pkl"` is a path to a `stable_baselines` model. (The environments used in the ICML paper have default values for `anchor`, so you do not have to specify one.) After running, let's say the best expression has traversal `["add", "x1", "x2"]`. To launch the second round of DSO, update the config's `action_spec` to use the fixed symbolic policy for the first action, learn a symbolic policy for the second action, and use the anchor again for the third action:
```
"action_spec" : [["add", "x1", "x2"], null, "anchor"]
```
After running DSO, say the second action's traversal is ["div", "x3", "x4"]. Finally, update the `action_spec` to:
```
"action_spec" : [["add", "x1", "x2"], ["div", "x3", "x4"], null]
```
and rerun DSO. The final result is a fully symbolic policy.

##### Configuring runs for learning decision tree policies

DSO can also be configured to learn a decision tree policy.
This is done by specifying `decision_tree_threshold_set` in `"task"`, which is a set of thresholds on the values of state variables when making a decision.
In particular, for each threshold `tj` in `decision_tree_threshold_set`, `StateChecker` tokens `xi < tj` for all
state variables `xi` will be added to the `Library`.

For example, for `MountainCarContinuous-v0`, here is an example config:
```
{
  "task" : {
    "task_type" : "control",
    "env" : "MountainCarContinuous-v0",
    "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 1.0, 5.0, 10.0]
    "decision_tree_threshold_set" : [-0.05, 0.0, 0.01]
  }
}
```
Other than the functions specified in `function_set`, this will also add `x1 < -0.05`, `x1 < 0.0`, `x1 < 0.01`, `x2 < -0.05`, `x2 < 0.0`, and `x2 < 0.01`
to the `Library` because `MountainCarContinuous-v0` has two state variables.
With these `StateChecker` tokens, decision tree policies like "if `x1` < -0.05 and `x2` < 0.0, the action is `exp(x1) + 1.0`; otherwise, the action is `sin(10 * x2)`" can be sampled.

### Using the Neural-Guided Genetic Programming Population Seeding Controller

To use the latest development from NeurIPS 2021, insert a field for `"gp_meld"`. You can play with the different parameters. The most important part is to set `"run_gp_meld"` to true.

```
{
  "gp_meld" : {
    "run_gp_meld" : true,
    "verbose" : false,
    "generations" : 20,
    "p_crossover" : 0.5,
    "p_mutate" : 0.5,
    "tournament_size" : 5,
    "train_n" : 50,
    "mutate_tree_max" : 3,
    "parallel_eval" : true
  }
}
```

# Sklearn interface

The `regression` task supports an additional [`sklearn`-like regressor interface](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html) to make it easy to try out deep symbolic regression on your own data:
```
from dso import DeepSymbolicRegressor

# Generate some data
np.random.seed(0)
X = np.random.random((10, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2

# Create the model
model = DeepSymbolicRegressor() # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit(X, y) # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions
model.predict(2 * X)

```

# Analyzing results

Each run of DSO saves a timestamped log directory in `config["training"]["logdir"]`. Inside this directory is:
* `dso_ExperimentName_0.csv`: This file contains batch-wise summary statistics for each epoch. The suffix `_0` means the random number seed was 0. (See "Advanced usage" for batch runs with multiple seeds.)
* `dso_ExperimnetName_0_summary.csv`: This file contains summary statistics for the entire training run.
* `dso_ExperimnetName_0_hof.csv`: This file contains statistics of the "hall of fame" (best sequences discovered during training). Edit `config["training"]["hof"] to set the number of hall-of-famers to record.
* `dso_ExperimnetName_0_pf.csv`: This file contains statistics of the Pareto front of sequences discovered during training. This is a reward-complexity front.
* `config.json`: This is a "dense" version of the configuration used for your run. It explicitly includes all parameters.

# Advanced usage

## Batch runs
DSO's command-line interface supports a `multiprocessing`-parallelized batch mode to run multiple tasks in parallel. This is recommended for large runs. Batch-mode DSO is launched with:

```
python -m dso.run path/to/config.json [--runs] [--n_cores_task] [--b] [--seed]
```

The option `--runs` (default `1`) defines how many independent tasks (with different random number seeds) to perform. The `regression` task is computationally expedient enough to run multiple tasks in parallel. For the `control` task, we recommend running with the default `--runs=1`.

The option `--n_cores_task` (default `1`) defines how many parallel processes to use across the `--runs` tasks. Each task is assigned a single core, so `--n_cores_task` should be less than or equal to `--runs`. (To use multiple cores _within_ a single task, i.e. to parallelize reward computation, see the `n_cores_batch` configuration parameter.)

The option `--seed`, *if provided*, will override the parameter `"seed"` in your config.

By default, DSO will use the task specification found in the configuration JSON. The option `--b` (default `None`) is used to specify the named task(s) via command-line. For example, `--b=path/to/mydata.csv` runs DSO on the given dataset (`regression` task), and `--b=MountainCarContinuous-v0` runs the environment MountainCarContinuous-v0 (`control` task). This is useful for running benchmark problems.

For example, to train 100 independent runs of DSR on the Nguyen-1 benchmark using 12 cores, using seeds 500 through 599:
```
python -m dso.run --b=Nguyen-1 --runs=100 --n_cores_task=12 --seed=500
```

## Simultaneously running multiple action dimensions on the `control` task
In your config file, you can set `"multiobject" : true` in the `task` section when you are running on `control`. This will completely ignore the `action_spec` in the config file and will instead attempt to learn all of the action dimensions at the same time. We are still improving this feature, so right now it has the following limits:

- It is "all-or-nothing", meaning you cannot mix and match your `action_spec` to learn a subset of the actions together and use the anchor or symbolic actions for the rest. Specifically, you can't do `"action_spec": [null, "anchor", null, "anchor"]` and just learn the two `null` actions simultaneously. Instead, the entire `action_spec` is ignored and all 4 actions are learned. We may add this capability in the future.
- One controller sequentially samples all tokens for all actions. This means that one RNN is sampling each action in order. This seems like its not ideal, given that RNNs suffer from catastrophic forgetting when sequences grow long enough. In the future, we will work on solutions to this problem.
- When you are training in a 1 action environment and set `multiobject = True`, `multiobject` is ignored and training progresses as normal.

## Adding custom tasks and priors

DSO supports adding custom tasks and priors from your own modules.

To add new tasks, the `task_type` keyword in the config file can be used in the following format: `<module>.<source>:<function>` specifying the source implementing a `make_task` function.

For example:
```
{
  "task" : {
    "task_type" : "custom_mod.my_source:make_task"
    }
  }
}
```

Similarly, new priors can be added by specifying the source where the `Prior` class can be found in the `prior` group of the config file.

For example:
```
 "prior": {
      "uniform_arity" : {
         "on" : true
      },
      "custom_mod.my_source:CustomPrior" : {
         "loc" : 10,
         "scale" : 5,
         "on" : true
      }
  }
```

# Citing this work

To cite this work, please cite according to the most relevant tasks and/or methods.

To cite the `regression` task, use:
```
@inproceedings{petersen2021deep,
  title={Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients},
  author={Petersen, Brenden K and Landajuela, Mikel and Mundhenk, T Nathan and Santiago, Claudio P and Kim, Soo K and Kim, Joanne T},
  booktitle={Proc. of the International Conference on Learning Representations},
  year={2021}
}
```

To cite the `control` task, use:
```
@inproceedings{landajuela2021discovering,
  title={Discovering symbolic policies with deep reinforcement learning},
  author={Landajuela, Mikel and Petersen, Brenden K and Kim, Sookyung and Santiago, Claudio P and Glatt, Ruben and Mundhenk, Nathan and Pettit, Jacob F and Faissol, Daniel},
  booktitle={International Conference on Machine Learning},
  pages={5979--5989},
  year={2021},
  organization={PMLR}
}
```

To cite the neural-guided genetic programming population seeding method, use:
```
@inproceedings{mundhenk2021seeding,
  title={Symbolic Regression via Neural-Guided Genetic Programming Population Seeding},
  author={T. Nathan Mundhenk and Mikel Landajuela and Ruben Glatt and Claudio P. Santiago and Daniel M. Faissol and Brenden K. Petersen},
  booktitle={35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
  year={2021}
}
```

# Release

LLNL-CODE-647188
