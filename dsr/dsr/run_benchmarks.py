import os
import json
import time
import multiprocessing
from functools import partial

import click
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr
from gplearn.genetic import SymbolicRegressor

from dsr.program import Program
from dsr.dataset import Dataset


def train_dsr(name, config_dataset, config_controller, config_training):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    try:
        import tensorflow as tf
        from dsr.controller import Controller
        from dsr.train import learn

    except:
        pass

    start = time.time()

    # Rename the output file
    config_training["output_file"] = name + ".csv"

    # Create the dataset
    X, y = get_dataset(name, config_dataset)
    Program.set_training_data(X, y)

    # Define the library
    Program.set_library(config_dataset["operators"], X.shape[1])

    tf.reset_default_graph()
    with tf.Session() as sess:        

        # Instantiate the controller
        controller = Controller(sess, **config_controller)

        # Train the controller
        result = learn(sess, controller, **config_training) # Reward, expression, traversal
        result["name"] = name
        result["t"] = time.time() - start
        return result


def train_gp(name, config_dataset, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    start = time.time()

    # Create the dataset
    X, y = get_dataset(name, config_dataset)

    # Configure parameters
    if "init_depth" in config_gp:
        config_gp["init_depth"] = tuple(config_gp["init_depth"])
    # config_gp["verbose"] = 0 # Turn off printing

    # Create the GP
    gp = SymbolicRegressor(**config_gp)

    # Fit the GP
    gp.fit(X, y)

    # Retrieve best results
    r = gp._program.fitness_
    base_r = gp._program.raw_fitness_
    
    # Currently outputting seralized program in place of its corresponding traversal
    program = str(gp._program)

    # Many failure cases right now for converting to SymPy expression...not high priority to fix
        # To do: serialized program --> tree --> SymPy-compatible tree --> traversal --> SymPy expression
    try:
        expression = repr(parse_expr(str(gp._program).replace("X", "x").replace("add", "Add").replace("mul", "Mul")))
    except:
        expression = "N/A"

    result = {
            "name" : name,
            "r" : r,
            "base_r" : base_r,
            "expression" : expression,
            "traversal" : program,
            "t" : time.time() - start
    }
    return result


def get_dataset(name, config_dataset):
    """Creates and returns the dataset"""

    config_dataset["name"] = name
    dataset = Dataset(**config_dataset)
    X, y = dataset.X_train, dataset.y_train
    return X, y


@click.command()
@click.argument('config_template', default="config.json")
@click.option('--method', default="dsr", type=click.Choice(["dsr", "gp"]), help="Symbolic regression method")
@click.option('--output_filename', default=None, help="Filename to write results")
@click.option('--num_cores', default=multiprocessing.cpu_count(), help="Number of cores to use")
@click.option('--exclude_fp_constants', is_flag=True, help="Exclude benchmark expressions containing floating point constants")
@click.option('--exclude_int_constants', is_flag=True, help="Exclude benchmark expressions containing integer constants")
@click.option('--exclude', multiple=True, type=str, help="Exclude benchmark expressions containing these names")
@click.option('--only', multiple=True, type=str, help="Only include benchmark expressions containing these names (overrides other exclusions)")
def main(config_template, method, output_filename, num_cores,
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

    if num_cores > len(names):
        print("Setting 'num_cores' to {} for batch because there are only {} expressions.".format(len(names), len(names)))
        num_cores = len(names)
    if config_training["verbose"] and num_cores > 1:
        print("Setting 'verbose' to False for parallelized run.")
        config_training["verbose"] = False
    if config_training["num_cores"] != 1 and num_cores > 1:
        print("Setting 'num_cores' to 1 for training (i.e. constant optimization) to avoid nested child processes.")
        config_training["num_cores"] = 1
    print("Running {} on benchmarks {}".format(method, names))

    # Define the work
    if method == "dsr":
        work = partial(train_dsr, config_dataset=config_dataset, config_controller=config_controller, config_training=config_training)
    elif method == "gp":
        work = partial(train_gp, config_dataset=config_dataset, config_gp=config_gp)

    # Farm out the work
    columns = ["name", "t", "base_r", "r", "expression", "traversal"]
    pd.DataFrame(columns=columns).to_csv(output_filename, header=True, index=False)
    if num_cores > 1:
        pool = multiprocessing.Pool(num_cores)    
        for result in pool.imap_unordered(work, names):
            pd.DataFrame(result, columns=columns, index=[0]).to_csv(output_filename, header=None, mode = 'a', index=False)
            print("Completed {} in {:.0f} s".format(result["name"], result["t"]))
    else:
        for name in names:
            result = work(name)
            pd.DataFrame(result, columns=columns, index=[0]).to_csv(output_filename, header=None, mode = 'a', index=False)


if __name__ == "__main__":
    main()
