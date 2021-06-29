"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import copy
import os
import time
import multiprocessing
from functools import partial
import zlib

import click
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr

from dsr import DeepSymbolicOptimizer
from dsr.task.regression.dataset import BenchmarkDataset
from dsr.baselines import gpsr
from dsr.logeval import LogEval
from dsr.config import load_config, set_benchmark_configs


def train_dsr(seeded_benchmark):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Override the benchmark name and output file
    benchmark_name, seed, config = seeded_benchmark
    print("benchmark name", benchmark_name)
    #config["task"]["name"] = benchmark_name
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

    run_config = copy.deepcopy(config)
    # Train the model
    model = DeepSymbolicOptimizer(run_config)
    start = time.time()
    result = {"name" : benchmark_name, "seed" : seed} # Name and seed are listed first
    result.update(model.train(seed=seed))
    result["t"] = time.time() - start
    result.pop("program")

    return result, run_config["paths"]["summary_path"]


def train_gp(seeded_benchmark): #, logdir, config_task, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    benchmark_name, seed, config = seeded_benchmark
    config_gp = config["gp"]
    config_gp["seed"] = seed + zlib.adler32(benchmark_name.encode("utf-8"))

    start = time.time()

    # Load the dataset
    config_dataset = config["task"]["dataset"]
    config_dataset["name"] = benchmark_name
    dataset = BenchmarkDataset(**config_dataset)

    # Fit the GP
    gp = gpsr.GP(dataset=dataset, **config_gp)
    p, logbook = gp.train()

    # Retrieve results
    r = p.fitness.values[0]
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
    df.to_csv(os.path.join(config["paths"]["log_dir"], "gp_{}_{}.csv".format(benchmark_name, seed)), index=False)

    result = {
        "name" : benchmark_name,
        "seed" : seed,
        "r" : r,
        "nmse_test" : nmse_test,
        "nmse_test_noiseless" : nmse_test_noiseless,
        "success" : success,
        "expression" : expression,
        "traversal" : str_p,
        "t" : time.time() - start
    }

    return result, config["paths"]["summary_path"]


@click.command()
@click.argument('config_template', default="")
@click.option('--method', default="dsr", type=click.Choice(["dsr", "gp"]), help="Symbolic regression method")
@click.option('--mc', default=None, help="Number of Monte Carlo trials for each benchmark")
@click.option('--output_filename', default=None, help="Filename to write results")
@click.option('--n_cores_task', '--n', default=1, help="Number of cores to spread out across tasks")
@click.option('--seed', default=None, help="First seed when running multiple experiments (increments by 1 for following experiments)")
@click.option('--b', multiple=True, type=str, help="Name of benchmark or benchmark prefix")
def main(config_template, method, mc, output_filename, n_cores_task, seed, b):
    """Runs DSR or GP on multiple benchmarks using multiprocessing."""

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template, method, mc)
    mc = config["task"]["runs"]

    # Set seed properly
    config["task"]["seed"] = config["task"]["seed"] if seed is None else int(seed)
    seed = config["task"]["seed"]

    # Load all benchmarks
    unique_benchmark_configs = set_benchmark_configs(config, b, method, output_filename)
    print("unique benchmark configs", unique_benchmark_configs)

    # Generate seeds for each run for each benchmark
    configs = []
    benchmarks = []
    seeds = []
    for benchmark in unique_benchmark_configs:
        benchmarks.extend([benchmark] * mc)
        configs.extend([unique_benchmark_configs[benchmark]] * mc)
        seeds.extend((np.arange(mc) + seed).tolist())
    seeded_benchmarks = list(zip(benchmarks, seeds, configs))
    benchmark_count = len(seeded_benchmarks)

    # Edit n_cores_task and/or n_cores_batch
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > benchmark_count:
        print("Setting 'n_cores_task' to {} for batch because there are only {} benchmark runs.".format(benchmark_count, benchmark_count))
        n_cores_task = benchmark_count
    if method == "dsr":
        for seeded_benchmark in seeded_benchmarks:
            if seeded_benchmark[2]["training"]["verbose"] and n_cores_task > 1:
                print("Setting 'verbose' to False for parallelized run.")
                seeded_benchmark[2]["training"]["verbose"] = False
            if seeded_benchmark[2]["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
                print("Setting 'n_cores_batch' to 1 to avoid nested child processes.")
                seeded_benchmark[2]["training"]["n_cores_batch"] = 1

    # Start benchmark training
    print("Running {} with {} seeds starting at {} on benchmark {}".format(method, mc, seed, [*unique_benchmark_configs]))

    # Define the work
    if method == "dsr":
        work = partial(train_dsr)
    elif method == "gp":
        work = partial(train_gp)

    # Farm out the work
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for result, summary_path in pool.imap_unordered(work, seeded_benchmarks):
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            print("\n  Completed {} seed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"], result["seed"] + 1 - seed, mc, result["t"]))
            print("########################################")
    else:
        for seeded_benchmark in seeded_benchmarks:
            result, summary_path = work(seeded_benchmark)
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            print("\n  Completed {} seed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"], result["seed"] + 1 - seed, mc, result["t"]))
            print("########################################")

    # Evaluate the log files
    for config in unique_benchmark_configs.values():
        log = LogEval(
            config["paths"]["log_dir"],
            config_file=config["paths"]["config_file"])
        log.analyze_log(
            show_count=config["postprocess"]["show_count"],
            show_hof=config["training"]["hof"] != None and config["training"]["hof"] > 0,
            show_pf=config["training"]["save_pareto_front"],
            save_plots=config["postprocess"]["save_plots"])


if __name__ == "__main__":
    main()