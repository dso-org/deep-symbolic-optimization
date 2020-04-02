"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

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
from dsr.dataset import Dataset
from dsr.baselines import gpsr
from dsr.train import learn
from dsr.language_model import LanguageModelPrior

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_dsr(name_and_seed, config_dataset, config_controller, config_language_model_prior, config_training):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    name, seed = name_and_seed

    try:
        import tensorflow as tf
        from dsr.controller import Controller
        from dsr.train import learn

        # Ignore TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    except:
        pass

    start = time.time()

    # Rename the output file
    config_training["output_file"] = "dsr_{}_{}.csv".format(name, seed)
    

    # Define the dataset, dsp parameters and library
    dataset = get_dataset(name, config_dataset)
    Program.clear_cache()
    Program.set_training_data(dataset)
    if "env_params" in config_training:
        Program.set_env_params(config_training)
    Program.set_library(dataset.function_set, dataset.n_input_var)
    if "env_params" in config_training:
        Program.set_action_params(config_training)
    Program.set_execute()

    tf.reset_default_graph()

    # Shift actual seed by checksum to ensure it's different across different benchmarks
    tf.set_random_seed(seed + zlib.adler32(name.encode("utf-8")))
  
    with tf.Session() as sess:

        # Instantiate the controller w/ language model
        if config_controller["use_language_model_prior"] and config_language_model_prior is not None:
            language_model_prior = LanguageModelPrior(dataset.function_set, dataset.n_input_var, **config_language_model_prior)
        else:
            language_model_prior = None
        controller = Controller(sess, debug=config_training["debug"], summary=config_training["summary"], language_model_prior=language_model_prior, **config_controller)

        # Train the controller
        result = learn(sess, controller, **config_training) # r, base_r, expression, traversal
        result["name"] = name
        result["t"] = time.time() - start

        result["seed"] = seed

        return result


def train_gp(name_and_seed, logdir, config_dataset, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    name, seed = name_and_seed
    config_gp["seed"] = seed + zlib.adler32(name.encode("utf-8"))

    start = time.time()

    # Load the dataset
    dataset = get_dataset(name, config_dataset)

    # Fit the GP
    gp = gpsr.GP(dataset=dataset, **config_gp)
    p, logbook = gp.train()

    # Retrieve results
    r = base_r = p.fitness.values[0]
    r_test = base_r_test = gp.eval_test(p)[0]
    str_p = str(p)
    nmse = gp.nmse(p)
    r_noiseless = base_r_noiseless = gp.eval_train_noiseless(p)[0]
    r_test_noiseless = base_r_test_noiseless = gp.eval_test_noiseless(p)[0]

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
        "nmse" : nmse,
        "r" : r,
        "base_r" : base_r,
        "r_test" : r_test,
        "base_r_test" : base_r_test,
        "r_noiseless" : r_noiseless,
        "base_r_noiseless" : base_r_noiseless,
        "r_test_noiseless" : r_test_noiseless,
        "base_r_test_noiseless" : base_r_test_noiseless,
        "expression" : expression,
        "traversal" : str_p,
        "t" : time.time() - start,
        "seed" : seed
    }

    return result


def get_dataset(name, config_dataset):
    """Creates and returns the dataset"""

    config_dataset["name"] = name
    dataset = Dataset(**config_dataset)
    return dataset


@click.command()
@click.argument('config_template', default="config.json")
@click.option('--method', default="dsr", type=click.Choice(["dsr", "gp"]), help="Symbolic regression method")
@click.option('--mc', default=1, type=int, help="Number of Monte Carlo trials for each benchmark")
@click.option('--output_filename', default=None, help="Filename to write results")
@click.option('--num_cores', default=multiprocessing.cpu_count(), help="Number of cores to use")
@click.option('--seed_shift', default=0, type=int, help="Integer to add to each seed (i.e. to combine multiple runs)")
@click.option('--benchmark', '--b', '--only', multiple=True, type=str, help="Benchmark or benchmark prefix to include")
def main(config_template, method, mc, output_filename, num_cores, seed_shift, benchmark):
    """Runs DSR or GP on multiple benchmarks using multiprocessing."""

     # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]              # Problem specification parameters
    config_training = config["training"]            # Training hyperparameters
    if "controller" in config:
        config_controller = config["controller"]    # Controller hyperparameters
    if "language_model_prior" in config:
        config_language_model_prior = config["language_model_prior"]            # Language model hyperparameters
    else:
        config_language_model_prior = None
    if "gp" in config:
        config_gp = config["gp"]                    # GP hyperparameters

    # Create output directories
    if output_filename is None:
        output_filename = "benchmark_{}.csv".format(method)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config_training["logdir"] += "_" + timestamp
    logdir = config_training["logdir"]
    os.makedirs(logdir, exist_ok=True)
    output_filename = os.path.join(logdir, output_filename)

    # Load the benchmark names
    data_path = resource_filename("dsr", "data/")
    benchmark_path = os.path.join(data_path, config_dataset["file"])
    df = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
    names = df["name"].to_list()

    # Load raw dataset names
    # HACK: Exclude "benchmark" names
    for f in os.listdir(data_path):
        if f.endswith(".csv") and "benchmarks" not in f and "function_sets" not in f:
            names.append(f.split('.')[0])

    # Load raw dataset from external directory in config
    if "extra_data_dir" in config_dataset:
        if not config_dataset["extra_data_dir"] == None:
            for f in os.listdir(config_dataset["extra_data_dir"]):
                if f.endswith(".csv"):
                    names.append(f.split('.')[0])

    # Filter out expressions
    expressions = [parse_expr(e) for e in df["sympy"]]
    if len(benchmark) > 0:
        keep = [False]*len(names)
        for included_name in benchmark:
            if '-' in included_name:
                keep = [True if included_name == n else k for k,n in zip(keep, names)]
            else:
                keep = [True if n.startswith(included_name) else k for k,n in zip(keep, names)]

    names = [n for k,n in zip(keep, names) if k]
    unique_names = names.copy()
    names *= mc

    # When passed to RNGs, these seeds will actually be added to checksums on the name
    seeds = (np.arange(mc) + seed_shift).repeat(len(unique_names)).tolist()
    names_and_seeds = list(zip(names, seeds))

    if num_cores > len(names):
        print("Setting 'num_cores' to {} for batch because there are only {} expressions.".format(len(names), len(names)))
        num_cores = len(names)
    if method == "dsr":
        if config_training["verbose"] and num_cores > 1:
            print("Setting 'verbose' to False for parallelized run.")
            config_training["verbose"] = False
        if config_training["num_cores"] != 1 and num_cores > 1:
            print("Setting 'num_cores' to 1 for training (i.e. constant optimization) to avoid nested child processes.")
            config_training["num_cores"] = 1
    print("Running {} for n={} on benchmarks {}".format(method, mc, unique_names))

    # Write terminal command and config.json into log directory
    cmd_filename = os.path.join(logdir, "cmd.out")
    with open(cmd_filename, 'w') as f:
        print(" ".join(sys.argv), file=f)
    config_filename = os.path.join(logdir, "config.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    # Define the work
    if method == "dsr":
        work = partial(train_dsr, config_dataset=config_dataset, config_controller=config_controller, config_language_model_prior=config_language_model_prior, config_training=config_training)
    elif method == "gp":
        work = partial(train_gp, logdir=logdir, config_dataset=config_dataset, config_gp=config_gp)

    # Farm out the work
    columns = ["name", "nmse", "base_r", "r", "base_r_test", "r_test", "base_r_noiseless", "r_noiseless", "base_r_test_noiseless", "r_test_noiseless", "expression", "traversal", "t", "seed"]
    pd.DataFrame(columns=columns).to_csv(output_filename, header=True, index=False)
    if num_cores > 1:
        pool = multiprocessing.Pool(num_cores)
        for result in pool.imap_unordered(work, names_and_seeds):
            pd.DataFrame(result, columns=columns, index=[0]).to_csv(output_filename, header=None, mode = 'a', index=False)
            print("Completed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"]+1-seed_shift, mc, result["t"]))
    else:
        for name_and_seed in names_and_seeds:
            result = work(name_and_seed)
            pd.DataFrame(result, columns=columns, index=[0]).to_csv(output_filename, header=None, mode = 'a', index=False)


if __name__ == "__main__":
    main()
