"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import copy
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

from dsr import DeepSymbolicOptimizer
from dsr.task.regression.dataset import BenchmarkDataset
from dsr.baselines import gpsr
from dsr.logeval import LogEval


def train_dsr(seeded_benchmark):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Override the benchmark name and output file
    benchmark_name, seed, config = seeded_benchmark
    #config["task"]["name"] = benchmark_name
    config["training"]["output_file"] = "dsr_{}_{}.csv".format(benchmark_name, seed)

    # Try importing TensorFlow (with suppressed warnings), Controller, and learn
    # When parallelizing across tasks, these will already be imported, hence try/except
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        ###from dsr.controller import Controller
        ###from dsr.train import learn
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

    return result, config["paths"]["summary_path"]


def train_gp(seeded_benchmark):
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
    df.to_csv(os.path.join(config["paths"]["log_dir"], "gp_{}_{}.csv".format(benchmark_name, seed)), index=False)

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

    return result, config["paths"]["summary_path"]


def _set_benchmark_configs(arg_benchmark, config, method, output_filename):
    """Get all indivual benchmarks and generate their respective configs."""
    # Use benchmark name from config if not specified as command-line arg
    if len(arg_benchmark) == 0:
        if isinstance(config["task"]["name"], str):
            benchmarks = (config["task"]["name"],)
        elif isinstance(config["task"]["name"], list):
            benchmarks = tuple(config["task"]["name"])
    else:
        benchmarks = arg_benchmark
    original_benchmarks = list(benchmarks)

    # Get log folder naming
    if any("..." in benchmark for benchmark in original_benchmarks) \
            or len(original_benchmarks) > 1:
        log_appendix = "Mixed"
    else:
        log_appendix = original_benchmarks[0]

    #If summaries are being saved on "training", summaries must be turned on under controller
    if config["training"].get("save_summary", True):
        if not config["controller"].get("summary", False):
            print('WARNING: When config["training"]["save_summary"] is true or absent, config["controller"]["summary"] '
                  'must be true. The summary recording will be turned on.')
        config["controller"]["summary"] = True

    # set common paths
    paths = {}
    paths["log_dir"] = os.path.join(
        config["training"]["logdir"],
        "log_{}_{}".format(
            datetime.now().strftime("%Y-%m-%d-%H%M%S"),
            log_appendix))
    # Create log dir and save commandline arguments
    os.makedirs(paths["log_dir"], exist_ok=True)
    with open(os.path.join(paths["log_dir"], "cmd.out"), 'w') as f:
        print(" ".join(sys.argv), file=f)

    # Update config where necessary
    config["training"]["logdir"] = paths["log_dir"]
    config["postprocess"]["method"] = method

    benchmark_df = None
    if config["task"]["task_type"] == "regression":
        paths["root_dir"] = resource_filename("dsr.task", "regression") \
            if config["task"]["dataset"]["root"] is None \
                else config["task"]["dataset"]["root"]
        paths["benchmark_file"] = "benchmarks.csv" \
            if config["task"]["dataset"]["benchmark_source"] is None \
                else config["task"]["dataset"]["benchmark_source"]
        paths["tokenset_path"] = os.path.join(
            paths["root_dir"], "function_sets.csv")
        if "dataset" in config["task"] \
                and "backup" in config["task"]["dataset"] \
                and config["task"]["dataset"]["backup"]:
            config["task"]["dataset"]["logdir"] = paths["log_dir"]
        # load all available benchmarks
        benchmark_df = pd.read_csv(
            os.path.join(paths["root_dir"], paths["benchmark_file"]),
            index_col=None, encoding="ISO-8859-1")
        # load available token sets
        if config["task"]["function_set"] is None:
            tokenset_df = pd.read_csv(
                paths["tokenset_path"],
                index_col=None, encoding="ISO-8859-1")

    # Helper functions
    def _set_individual_paths(benchmark):
        new_paths = copy.deepcopy(paths)
        new_paths["config_file"] = "{}_{}_config.json".format(method, benchmark)
        new_paths["config_path"] = os.path.join(
            paths["log_dir"], new_paths["config_file"])
        if output_filename is None:
            new_paths["summary_path"] = os.path.join(
                paths["log_dir"], "{}_{}_summary.csv".format(method, benchmark))
        else:
            new_paths["summary_path"] = output_filename
        return new_paths

    def _set_individual_config(benchmark):
        new_config = copy.deepcopy(config)
        new_config["task"]["name"] = benchmark
        if not isinstance(config["task"]["function_set"], list):
            tokenset_name = benchmark_df[
                benchmark_df["name"]==benchmark]["function_set"].item()
            new_config["task"]["function_set"] = tokenset_df[
                tokenset_df["name"]==tokenset_name]["function_set"].item().split(',')
        new_config["paths"] = _set_individual_paths(benchmark)
        with open(new_config["paths"]["config_path"], 'w') as f:
            json.dump(new_config, f, indent=4)
        return new_config

    # make sure we get the right benchmarks
    benchmarks = {}
    for benchmark in original_benchmarks:
        if benchmark[-3:] == "...":
            benchmark_list = list(benchmark_df['name'].loc[benchmark_df['name'].str.startswith(benchmark[:-3])])
            for each_benchmark in benchmark_list:
                benchmarks[each_benchmark] = _set_individual_config(each_benchmark)
            continue
        benchmarks[benchmark] = _set_individual_config(benchmark)
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

    # Load all benchmarks
    unique_benchmark_configs = _set_benchmark_configs(b, config, method, output_filename)

    # Generate seeds for each run for each benchmark
    configs = []
    benchmarks = []
    seeds = []
    for benchmark in unique_benchmark_configs:
        benchmarks.extend([benchmark] * mc)
        configs.extend([unique_benchmark_configs[benchmark]] * mc)
        seeds.extend((np.arange(mc) + seed_shift).tolist())
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
    print("Running {} with {} seeds on benchmark {}".format(method, mc, [*unique_benchmark_configs]))

    # Define the work
    if method == "dsr":
        #work = partial(train_dsr, config=config)
        work = partial(train_dsr)
    elif method == "gp":
        assert config_task["task_type"] == "regression", \
            "Pure GP currently only supports the regression task."
        work = partial(train_gp)

    # Farm out the work
    write_header = True
    if n_cores_task > 1:
        pool = multiprocessing.Pool(n_cores_task)
        for result, summary_path in pool.imap_unordered(work, seeded_benchmarks):
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            print("Completed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"]+1-seed_shift, mc, result["t"]))
            write_header = False
    else:
        for seeded_benchmark in seeded_benchmarks:
            result, summary_path = work(seeded_benchmark)
            pd.DataFrame(result, index=[0]).to_csv(summary_path, header=not os.path.exists(summary_path), mode='a', index=False)
            write_header = False

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
