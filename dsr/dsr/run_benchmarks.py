import json
import multiprocessing
from functools import partial

import click
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr

from dsr.program import Program
from dsr.dataset import Dataset


def train_benchmark(name, config_dataset, config_controller, config_training):

    try:
        import tensorflow as tf
        from dsr.controller import Controller
        from dsr.train import learn

    except:
        pass

    # Create the dataset
    config_dataset["name"] = name
    dataset = Dataset(**config_dataset)
    X, y = dataset.X_train, dataset.y_train

    # Define the library
    Program.set_library(config_dataset["operators"], X.shape[1])
    n_choices = len(Program.library)

    # Turn off printing
    config_training["verbose"] = False    

    tf.reset_default_graph()
    with tf.Session() as sess:        

        # Instantiate the controller
        controller = Controller(sess, n_choices=n_choices, **config_controller)

        result = learn(sess, controller, X, y, **config_training) # Reward, expression, traversal
        result["name"] = name
        return result


@click.command()
@click.argument('config_filename', default="config.json")
@click.option('--output_filename', default="benchmark_results.csv", help="Filename to write results")
@click.option('--num_cores', default=multiprocessing.cpu_count(), help="Number of cores to use")
@click.option('--exclude_fp_constants', is_flag=True, help="Exclude benchmark expressions containing floating point constants")
@click.option('--exclude_int_constants', is_flag=True, help="Exclude benchmark expressions containing integer constants")
def main(config_filename, output_filename, num_cores, exclude_fp_constants, exclude_int_constants):
    
     # Load the config file
    with open(config_filename, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]          # Problem specification parameters
    config_training = config["training"]        # Training hyperparameters
    config_controller = config["controller"]    # Controller hyperparameters

    # Load the benchmark names
    df = pd.read_csv(config_dataset["file"], encoding="ISO-8859-1")
    names = df["name"].to_list()

    # Filter out expressions
    expressions = [parse_expr(e) for e in df["expression"]]
    keep = [True]*len(expressions)
    if exclude_fp_constants:
        keep = [False if "Float" in srepr(e) else k for k,e in zip(keep, expressions)]
    if exclude_int_constants:
        keep = [False if "Integer" in srepr(e) else k for k,e in zip(keep, expressions)]
    names = [n for k,n in zip(keep, names) if k]

    # Farm out the work
    pool = multiprocessing.Pool(num_cores)
    work = partial(train_benchmark, config_dataset=config_dataset, config_controller=config_controller, config_training=config_training)
    columns = ["name", "r", "expression", "traversal"]
    pd.DataFrame(columns=columns).to_csv(output_filename, header=True, index=False)
    for result in pool.imap_unordered(work, names):
        pd.DataFrame(result, columns=columns, index=[0]).to_csv(output_filename, header=None, mode = 'a', index=False)


if __name__ == "__main__":
    main()