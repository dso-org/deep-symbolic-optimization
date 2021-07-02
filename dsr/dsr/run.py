"""Parallelized, single-point launch script to run DSO on a set of benchmarks."""

import os
import time
import multiprocessing
from copy import deepcopy
from datetime import datetime

import click
import pandas as pd

from dsr import DeepSymbolicOptimizer
from dsr.logeval import LogEval
from dsr.config import load_config


def train_dsr(config):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Try importing TensorFlow (with suppressed warnings), Controller, and learn
    # When parallelizing across tasks, these will already be imported, hence try/except
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ModuleNotFoundError: # Specific subclass of ImportError for when module is not found, probably needs to be excepted first
        print("One or more libraries not found")
        raise ModuleNotFoundError
    except ImportError:
        # Have we already imported tf? If so, this is the error we want to dodge.
        if 'tf' in globals():
            pass
        else:
            raise ImportError

    # For some reason, for the control task, the environment needs to be instantiated
    # before creating the pool. Otherwise, gym.make() hangs during the pool initializer
    if config["task"]["task_type"] == "control" and config["training"]["n_cores_batch"] > 1:
        import gym
        import dsr.task.control # Registers custom and third-party environments
        gym.make(config["task"]["env"])

    # Train the model
    model = DeepSymbolicOptimizer(deepcopy(config))
    start = time.time()
    result = model.train()
    result["t"] = time.time() - start
    result.pop("program")

    return result


@click.command()
@click.argument('config_template', default="")
@click.option('--mc', default=1, type=int, help="Number of Monte Carlo trials for each benchmark")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--seed_shift', '--ss', default=0, type=int, help="Integer to add to each seed (i.e. to combine multiple runs)")
@click.option('--benchmark', '--b', default=None, type=str, help="Name of benchmark")
def main(config_template, mc, n_cores_task, seed_shift, benchmark):
    """Runs DSR or GP on multiple benchmarks using multiprocessing."""

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)

    # Overwrite benchmark (for tasks that support them)
    task_type = config["task"]["task_type"]
    if benchmark is not None:
        # For regression, --b overwrites config["task"]["dataset"]
        if task_type == "regression":
            config["task"]["dataset"] = benchmark
        # For regression, --b overwrites config["task"]["env"]
        elif task_type == "control":
            config["task"]["env"] = benchmark
        else:
            raise ValueError("--b is not supported for task {}.".format(task_type))

    # Provide default experiment name
    if config["experiment"]["exp_name"] is None:
        config["experiment"]["exp_name"] = task_type

    # Set save path: [logdir]/[exp_name]/[timestamp]
    save_path = os.path.join(
        config["experiment"]["logdir"],
        config["experiment"]["exp_name"],
        datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    config["experiment"]["save_path"] = save_path
    os.makedirs(save_path, exist_ok=False)
    summary_path = os.path.join(save_path, "summary.csv")

    # Fix incompatible configurations
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > mc:
        print("Setting 'n_cores_task' to {} because there are only {} replicates.".format(mc, mc))
        n_cores_task = mc
    if config["training"]["verbose"] and n_cores_task > 1:
        print("Setting 'verbose' to False for parallelized run.")
        config["training"]["verbose"] = False
    if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
        print("Setting 'n_cores_batch' to 1 to avoid nested child processes.")
        config["training"]["n_cores_batch"] = 1

    # Generate configs for each mc
    configs = [deepcopy(config) for _ in range(mc)]
    if config["experiment"]["seed"] is not None:
        for i, config in enumerate(configs):
            config["experiment"]["seed"] += seed_shift + i

    # Start benchmark training
    print("Running DSO for {} seeds".format(mc))

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for i, result in enumerate(pool.imap_unordered(train_dsr, configs)):
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            print("Completed {} of {} in {:.0f} s".format(i + 1, mc, result["t"]))
    else:
        for i, config in enumerate(configs):
            result = train_dsr(config)
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            print("Completed {} of {} in {:.0f} s".format(i + 1, mc, result["t"]))

    # Evaluate the log files
    log = LogEval(config_path=os.path.join(save_path, "config.json"))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["training"]["hof"] is not None and config["training"]["hof"] > 0,
        show_pf=config["training"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"])


if __name__ == "__main__":
    main()
