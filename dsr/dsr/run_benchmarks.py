import os
import json
import time
import multiprocessing
from functools import partial

import click
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr
from gplearn.genetic import SymbolicRegressor

from dsr.program import Program
from dsr.dataset import Dataset


def train_dsr(name_and_seed, config_dataset, config_controller, config_training):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    name, seed = name_and_seed

    try:
        import tensorflow as tf
        from dsr.controller import Controller
        from dsr.train import learn

    except:
        pass

    start = time.time()

    # Rename the output file
    config_training["output_file"] = name + ".csv"

    # Define the dataset and library
    dataset = get_dataset(name, config_dataset)
    Program.clear_cache()
    Program.set_training_data(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
    Program.set_library(dataset.function_set, dataset.n_input_var)

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    with tf.Session() as sess:        

        # Instantiate the controller
        controller = Controller(sess, summary=config_training["summary"], **config_controller)

        # Train the controller
        result = learn(sess, controller, **config_training) # r, base_r, expression, traversal
        result["name"] = name
        result["t"] = time.time() - start

        result["seed"] = seed

        return result


def train_gp(name_and_seed, logdir, config_dataset, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    name, seed = name_and_seed

    start = time.time()

    # Create the dataset
    dataset = get_dataset(name, config_dataset)

    # Configure parameters
    for to_be_tuple in ["init_depth", "const_range"]:
        if to_be_tuple in config_gp:
            config_gp[to_be_tuple] = tuple(config_gp[to_be_tuple])
    config_gp["function_set"] = dataset.function_set.copy()
    if "const" in config_gp["function_set"]:
        config_gp["function_set"].remove("const")
    config_gp["random_state"] = seed
    # config_gp["verbose"] = 0 # Turn off printing
    
    # Parameter assertions
    assert "const_range" in config_gp, "Constant range must be specified (or None if not using constants)."
    if config_gp["const_range"] is None:
        assert "const" not in dataset.function_set, "Constant range not provided but constants are in dataset."
    elif "const" not in dataset.function_set:
        print("Constant range provided but constant not in function set for benchmark {}. Overriding constant range to None.".format(name))
        config_gp["const_range"] = None    

    # Fit the GP
    gp = SymbolicRegressor(**config_gp)
    gp.fit(dataset.X_train, dataset.y_train)

    # Save run details
    # NOTE: gp._program contains the best program (in terms of raw reward) in
    # the final generation, not necessarily the best overall.
    df = pd.DataFrame(data=gp.run_details_)
    rename = {"best_fitness" : "base_r_max",
              "average_fitness" : "base_r_avg",
              "generation_time" : "t"
              }
    df = df.rename(columns=rename)
    df["base_r_best"] = df["base_r_max"].cummin()
    df.to_csv(os.path.join(logdir, name + "_gp.csv"))

    # Retrieve best program
    # gplearn does not store the best overall program, so the best may be N/A
    base_r = df["base_r_best"].min()
    index_of_best = np.argmin(df["base_r_best"].values)
    for p in gp._programs[index_of_best]:
        if p is not None and p.raw_fitness_ == base_r:
            break
    r = p.fitness_ if p is not None else "N/A"
    
    # Currently outputting seralized program in place of its corresponding traversal
    str_p = str(p) if p is not None else "N/A"

    # Many failure cases right now for converting to SymPy expression...not high priority to fix
    # To do: serialized program --> tree --> SymPy-compatible tree --> traversal --> SymPy expression
    try:
        expression = repr(parse_expr(str_p.replace("X", "x").replace("add", "Add").replace("mul", "Mul")))
    except:
        expression = "N/A"

    # Compute r and base_r on test set
    if p is not None:
        p.raw_fitness_ = p.raw_fitness(X=dataset.X_test, y=dataset.y_test, sample_weight=None) # Must override p.raw_fitness_ because p.fitness() uses it in its calculation
        base_r_test = p.raw_fitness_
        r_test = p.fitness(parsimony_coefficient=gp.parsimony_coefficient)

        y_hat_test = p.execute(dataset.X_test)
        var_y = np.var(dataset.y_train) # Always normalize using training data
        nmse = np.mean((dataset.y_test - y_hat_test)**2) / var_y
    else:
        base_r_test = "N/A"
        r_test = "N/A"
        nmse = "N/A"

    result = {
            "name" : name,
            "nmse" : nmse,
            "r" : r,
            "base_r" : base_r,
            "r_test" : r_test,
            "base_r_test" : base_r_test,
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
@click.option('--exclude_fp_constants', is_flag=True, help="Exclude benchmark expressions containing floating point constants")
@click.option('--exclude_int_constants', is_flag=True, help="Exclude benchmark expressions containing integer constants")
@click.option('--exclude', multiple=True, type=str, help="Exclude benchmark expressions containing these names")
@click.option('--only', multiple=True, type=str, help="Only include benchmark expressions containing these names (overrides other exclusions)")
def main(config_template, method, mc, output_filename, num_cores, seed_shift,
         exclude_fp_constants, exclude_int_constants, exclude, only):
    """Runs DSR or GP on multiple benchmarks using multiprocessing."""
    
     # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]          # Problem specification parameters
    config_training = config["training"]        # Training hyperparameters
    config_controller = config["controller"]    # Controller hyperparameters
    config_gp = config["gp"]                    # GP hyperparameters

    # Create output directories
    if output_filename is None:
        output_filename = "benchmark_{}.csv".format(method)
    logdir = os.path.join("log", config_training["logdir"])
    os.makedirs(logdir, exist_ok=True)
    output_filename = os.path.join(logdir, output_filename)

    # Load the benchmark names
    df = pd.read_csv(config_dataset["file"], encoding="ISO-8859-1")
    names = df["name"].to_list()

    # Filter out expressions
    expressions = [parse_expr(e) for e in df["expression"]]    
    if len(only) == 0:
        keep = [True]*len(expressions)
        if exclude_fp_constants:
            keep = [False if "Float" in srepr(e) else k for k,e in zip(keep, expressions)]
        if exclude_int_constants:
            keep = [False if "Integer" in srepr(e) else k for k,e in zip(keep, expressions)]
        for excluded_name in exclude:
            keep = [False if excluded_name in n else k for k,n in zip(keep, names)]
    else:
        keep = [False]*len(expressions)
        for included_name in only:
            if '-' in included_name: # If the whole name is specified (otherwise, e.g., only=Name-1 will also apply Name-10, Name-11, etc.)
                keep = [True if included_name == n else k for k,n in zip(keep, names)]
            else:
                keep = [True if included_name in n else k for k,n in zip(keep, names)]

    names = [n for k,n in zip(keep, names) if k]
    unique_names = names.copy()
    names *= mc
    seeds = (np.arange(mc) + seed_shift).repeat(len(unique_names)).tolist()
    names_and_seeds = list(zip(names, seeds)) # E.g. [("Koza-1", 0), ("Nguyen-1", 0), ("Koza-2", 1), ("Nguyen-1", 1)]

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
    elif method == "gp":
        if config_gp["verbose"] and num_cores > 1:
            print("Setting 'verbose' to False for parallelized run.")
            config_gp["verbose"] = False
        if config_gp["n_jobs"] != 1 and num_cores > 1:
            print("Setting 'n_jobs' to 1 for training to avoid nested child processes.")
            config_gp["n_jobs"] = 1
    print("Running {} for n={} on benchmarks {}".format(method, mc, unique_names))

    # Define the work
    if method == "dsr":
        work = partial(train_dsr, config_dataset=config_dataset, config_controller=config_controller, config_training=config_training)
    elif method == "gp":
        work = partial(train_gp, logdir=logdir, config_dataset=config_dataset, config_gp=config_gp.copy())

    # Farm out the work
    columns = ["name", "nmse", "base_r", "r", "base_r_test", "r_test", "expression", "traversal", "t", "seed"]
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
