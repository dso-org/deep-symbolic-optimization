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
from functools import partial
from pkg_resources import resource_filename
import zlib

import click
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr

from dsr import DeepSymbolicOptimizer
from dsr.program import Program
from dsr.task.regression.dataset import BenchmarkDataset
from dsr.baselines import gpsr


def train_dsr(seeded_benchmark, config):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Override the benchmark name and output file
    benchmark_name, seed = seeded_benchmark
    config["task"]["name"] = benchmark_name
    config["training"]["output_file"] = "dsr_{}_{}.csv".format(benchmark_name, seed)

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
    if config["task"]["task_type"] == "control" and config["training"]["n_cores_batch"] > 1:
        import gym
        gym.make(benchmark_name)

    # Train the model
    model = DeepSymbolicOptimizer(config)
    start = time.time()
    result = {"name" : benchmark_name, "seed" : seed} # Name and seed are listed first
    result.update(model.train(seed=seed))
    result["t"] = time.time() - start
    result.pop("program")

    return result


def train_gp(seeded_benchmark, logdir, config_task, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    benchmark_name, seed = seeded_benchmark
    config_gp["seed"] = seed + zlib.adler32(benchmark_name.encode("utf-8"))

    start = time.time()

    # Load the dataset
    config_dataset = config_task["dataset"]
    config_dataset["name"] = benchmark_name
    dataset = BenchmarkDataset(**config_dataset)

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
    df.to_csv(os.path.join(logdir, "gp_{}_{}.csv".format(benchmark_name, seed)), index=False)

    result = {
        "name" : benchmark_name,
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


def _get_benchmarks(arg_benchmark, config_benchmarks, benchmark_dir=None):
    # Use benchmark name from config if not specified as command-line arg
    if len(arg_benchmark) == 0:
        if isinstance(config_benchmarks, str):
            benchmarks = (config_benchmarks,)
        elif isinstance(config_benchmarks, list):
            benchmarks = tuple(config_benchmarks)
    else:
        benchmarks = arg_benchmark
    original_benchmarks = list(benchmarks)
    # load all available benchmarks
    benchmark_dir = resource_filename("dsr.task", "regression") if benchmark_dir == None else benchmark_dir
    benchmark_path = os.path.join(
        benchmark_dir,
        "benchmarks.csv")
    benchmark_df = pd.read_csv(benchmark_path, index_col=None, encoding="ISO-8859-1")
    # make sure we get the right benchmarks
    benchmarks = []
    for benchmark in original_benchmarks:
        if benchmark[-3:] == "...":
            replace_benchmarks = list(benchmark_df['name'].loc[benchmark_df['name'].str.startswith(benchmark[:-3])])
            benchmarks += replace_benchmarks
            continue
        benchmarks.append(benchmark)
    return benchmarks


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

    # Load all benchmarks
    unique_benchmarks = _get_benchmarks(b, config_task["name"], config_task["dataset"]["root"])
    # Generate benchmark-seed pairs for each MC. When passed to the TF RNG,
    # seeds will be added to checksums on the benchmark names
    benchmarks = unique_benchmarks.copy()

    benchmarks *= mc
    seeds = (np.arange(mc) + seed_shift).repeat(len(unique_benchmarks)).tolist()
    seeded_benchmarks = list(zip(benchmarks, seeds))
    benchmark_count = len(seeded_benchmarks)


    logfile_name = unique_benchmarks[0] if len(unique_benchmarks) == 1 else 'Mixed'
    # Create output directories
    if output_filename is None:
        output_filename = "{}_{}_summary.csv".format(method, logfile_name)

    config_training["logdir"] = os.path.join(
        config_training["logdir"],
        "log_{}_{}".format(datetime.now().strftime("%Y-%m-%d-%H%M%S"), logfile_name))

    logdir = config_training["logdir"]
    if "dataset" in config_task and "backup" in config_task["dataset"] and config_task["dataset"]["backup"]:
        config_task["dataset"]["logdir"] = logdir
    os.makedirs(logdir, exist_ok=True)
    output_filename = os.path.join(logdir, output_filename)

    # Edit n_cores_task and/or n_cores_batch
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > benchmark_count:
        print("Setting 'n_cores_task' to {} for batch because there are only {} benchmarks.".format(benchmark_count, benchmark_count))
        n_cores_task = benchmark_count
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
        work = partial(train_dsr, config=config)
    elif method == "gp":
        work = partial(train_gp, logdir=logdir, config_task=config_task, config_gp=config_gp)

    # Farm out the work
    write_header = True
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for result in pool.imap_unordered(work, seeded_benchmarks):
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            print("Completed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"]+1-seed_shift, mc, result["t"]))
            write_header = False
    else:
        for seeded_benchmark in seeded_benchmarks:
            result = work(seeded_benchmark)
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            write_header = False

    print("Results saved to: {}".format(output_filename))


if __name__ == "__main__":
    main()
