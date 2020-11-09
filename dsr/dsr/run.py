"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
import json
import time
from datetime import datetime
import multiprocessing
from copy import deepcopy
from functools import partial
from pkg_resources import resource_filename
import zlib

import click
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr

from dsr.program import Program
from dsr.task.regression.dataset import Dataset
from dsr.baselines import gpsr
from dsr.language_model import LanguageModelPrior
from dsr.task import set_task
import dsr.gp as gp_dsr


def train_dsr(name_and_seed, config_task, config_controller, config_language_model_prior, config_training, config_gp_meld):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Override the benchmark name
    name, seed = name_and_seed
    config_task["name"] = name

    # Try importing TensorFlow (with suppressed warnings), Controller, and learn
    # When parallelizing across tasks, these will already be imported, hence try/except
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        from dsr.controller import Controller
        from dsr.train import learn
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
    if config_task["task_type"] == "control" and config_training["n_cores_batch"] > 1:
        import gym
        gym.make(name)

    # Create the pool and set the task for each worker
    n_cores_batch = config_training["n_cores_batch"]
    if n_cores_batch > 1:
        pool = multiprocessing.Pool(n_cores_batch, initializer=set_task, initargs=(config_task,))
    else:
        pool = None

    # Set the task for the parent process
    set_task(config_task)

    start = time.time()

    # Rename the output file
    config_training["output_file"] = "dsr_{}_{}.csv".format(name, seed)

    # Reset cache and TensorFlow graph
    Program.clear_cache()
    tf.reset_default_graph()

    # Shift actual seed by checksum to ensure it's different across different benchmarks
    tf.set_random_seed(seed + zlib.adler32(name.encode("utf-8")))

    with tf.Session() as sess:

        # Instantiate the controller w/ language model
        if config_controller["use_language_model_prior"] and config_language_model_prior is not None:
            language_model_prior = LanguageModelPrior(function_set, n_input_var, **config_language_model_prior)
        else:
            language_model_prior = None
        controller = Controller(sess, debug=config_training["debug"], summary=config_training["summary"], language_model_prior=language_model_prior, **config_controller)

        if config_gp_meld is not None and config_gp_meld["run_gp_meld"]:
            gp_controller           = gp_dsr.GPController(config_gp_meld, config_task, config_training)
        else:
            gp_controller           = None

        # Train the controller
        result = {"name" : name, "seed" : seed} # Name and seed are listed first
        result.update(learn(sess, controller, pool, gp_controller, **config_training))
        result["t"] = time.time() - start # Time listed last

        return result



def train_gp(name_and_seed, logdir, config_task, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    name, seed = name_and_seed
    config_gp["seed"] = seed + zlib.adler32(name.encode("utf-8"))

    start = time.time()

    # Load the dataset
    config_dataset = config_task["dataset"]
    config_dataset["name"] = name
    dataset = Dataset(**config_dataset)

    # Fit the GP
    gp = gpsr.GP(dataset=dataset, **config_gp)
    p, logbook = gp.train()

    # Retrieve results
    r = base_r = p.fitness.values[0]
    str_p = str(p)
    nmse_test = gp.nmse_test(p)[0]
    nmse_test_noiseless = gp.nmse_test_noiseless(p)[0]
    success = gp.success(p)

    # Many failure cases right now for converting to SymPy expression
    try:
        expression = repr(parse_expr(str_p.replace("X", "x").replace("add", "Add").replace("mul", "Mul")))
    except:
        expression = "N/A"

    # Save run details
    drop = ["gen", "nevals"]
    df_fitness = pd.DataFrame(logbook.chapters["fitness"]).drop(drop, axis=1)
    df_fitness = df_fitness.rename({"avg" : "fit_avg", "min" : "fit_min"}, axis=1)
    df_fitness["fit_best"] = df_fitness["fit_min"].cummin()
    df_len = pd.DataFrame(logbook.chapters["size"]).drop(drop, axis=1)
    df_len = df_len.rename({"avg" : "l_avg"}, axis=1)
    df = pd.concat([df_fitness, df_len], axis=1, sort=False)
    df.to_csv(os.path.join(logdir, "gp_{}_{}.csv".format(name, seed)), index=False)

    result = {
        "name" : name,
        "seed" : seed,
        "r" : r,
        "base_r" : base_r,
        "nmse_test" : nmse_test,
        "nmse_test_noiseless" : nmse_test_noiseless,
        "success" : success,
        "expression" : expression,
        "traversal" : str_p,
        "t" : time.time() - start
    }

    return result


@click.command()
@click.argument('config_template', default="config.json")
@click.option('--method', default="dsr", type=click.Choice(["dsr", "gp"]), help="Symbolic regression method")
@click.option('--mc', default=1, type=int, help="Number of Monte Carlo trials for each benchmark")
@click.option('--output_filename', default=None, help="Filename to write results")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--seed_shift', default=0, type=int, help="Integer to add to each seed (i.e. to combine multiple runs)")
@click.option('--b', multiple=True, type=str, help="Name of benchmark or benchmark prefix")
def main(config_template, method, mc, output_filename, n_cores_task, seed_shift, b):
    """Runs DSR or GP on multiple benchmarks using multiprocessing."""

    # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)

    # Required configs
    config_task = config["task"]            # Task specification parameters
    config_training = config["training"]    # Training hyperparameters

    # Optional configs
    config_controller = config.get("controller")                        # Controller hyperparameters
    config_language_model_prior = config.get("language_model_prior")    # Language model hyperparameters
    config_gp = config.get("gp")                                        # GP hyperparameters
    config_gp_meld = config.get('gp_meld')

    # Create output directories
    if output_filename is None:
        output_filename = "benchmark_{}.csv".format(method)
    config_training["logdir"] = os.path.join(
        config_training["logdir"],
        "log_{}".format(datetime.now().strftime("%Y-%m-%d-%H%M%S")))
    logdir = config_training["logdir"]
    if "dataset" in config_task and "backup" in config_task["dataset"] and config_task["dataset"]["backup"]:
        config_task["dataset"]["logdir"] = logdir
    os.makedirs(logdir, exist_ok=True)
    output_filename = os.path.join(logdir, output_filename)
    # Use benchmark name from config if not specified as command-line arg
    if len(b) == 0:
        if isinstance(config_task["name"], str):
            b = (config_task["name"],)
        elif isinstance(config_task["name"], list):
            b = tuple(config_task["name"])

    # HACK: DSR-specific shortcut to run all Nguyen benchmarks
    benchmarks = list(b)
    if "Nguyen" in benchmarks:
        benchmarks.remove("Nguyen")
        benchmarks += ["Nguyen-{}".format(i+1) for i in range(12)]

    # Generate benchmark-seed pairs for each MC. When passed to the TF RNG,
    # seeds will be added to checksums on the benchmark names
    unique_benchmarks = benchmarks.copy()
    benchmarks *= mc
    seeds = (np.arange(mc) + seed_shift).repeat(len(unique_benchmarks)).tolist()
    names_and_seeds = list(zip(benchmarks, seeds))

    # Edit n_cores_task and/or n_cores_batch
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > len(benchmarks):
        print("Setting 'n_cores_task' to {} for batch because there are only {} benchmarks.".format(len(benchmarks), len(benchmarks)))
        n_cores_task = len(benchmarks)
    if method == "dsr":
        if config_training["verbose"] and n_cores_task > 1:
            print("Setting 'verbose' to False for parallelized run.")
            config_training["verbose"] = False
        if config_training["n_cores_batch"] != 1 and n_cores_task > 1:
            print("Setting 'n_cores_batch' to 1 to avoid nested child processes.")
            config_training["n_cores_batch"] = 1
    print("Running {} for n={} on benchmarks {}".format(method, mc, unique_benchmarks))

    # Write terminal command and config.json into log directory
    cmd_filename = os.path.join(logdir, "cmd.out")
    with open(cmd_filename, 'w') as f:
        print(" ".join(sys.argv), file=f)
    config_filename = os.path.join(logdir, "config.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    # Define the work
    if method == "dsr":
        work = partial(train_dsr, config_task=config_task, config_controller=config_controller, config_language_model_prior=config_language_model_prior, config_training=config_training, config_gp_meld=config_gp_meld)
    elif method == "gp":
        work = partial(train_gp, logdir=logdir, config_task=config_task, config_gp=config_gp)

    # Farm out the work
    write_header = True
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for result in pool.imap_unordered(work, names_and_seeds):
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            print("Completed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"]+1-seed_shift, mc, result["t"]))
            write_header = False
    else:
        for name_and_seed in names_and_seeds:
            result = work(name_and_seed)
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            write_header = False

    print("Results saved to: {}".format(output_filename))


if __name__ == "__main__":
    main()
