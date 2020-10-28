"""Class for deterministically generating a benchmark dataset from benchmark specifications."""

import os
import ast
import itertools
from textwrap import indent
from pkg_resources import resource_filename
import zlib

import click
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify, pretty, srepr

from dsr.functions import function_map


class Dataset(object):
    """
    Class used to generate X, y data from a named benchmark expression or raw
    dataset.

    The numpy expression is used to evaluate the expression using any custom/
    protected functions in function_map. The sympy expression is only used for
    printing, not function evaluation.

    Parameters
    ----------
    file : str
        Filename of CSV with benchmark expressions, contained in dsr/data.

    name : str
        Name of expression or raw dataset.

    seed : int, optional
        Random number seed used to generate data. Checksum on name is added to
        seed.

    noise : float, optional
        If not None, Gaussian noise is added to the y values with standard
        deviation = noise * RMS of the noiseless y training values.

    function_set : list, optional
        List of operators to use in library. If None, use default. Otherwise,
        this will override any default, e.g. when using a benchmark expression.

    extra_data_dir : str, optional
        Absolute path of directory to look for dataset files. If None, uses
        default path dsr/data.

    dataset_size_multiplier : float, optional
        Multiplier for size of the dataset. Only works for expressions.

    shuffle_data : bool, optional
        Shuffle the dataset. Only work for external datasets.

    train_fraction : float, optional
        Fraction of dataset used for training. Only work for external datasets.

    experiment_root : str, optional
        Directory where the files that are used for the generation of the
        dataset are stored.

    logdir : str, optional
        Directory where experiment logfiles are saved.

    backup : bool, optional
        Save generated or loaded dataset in logdir. Will only work if
        logdir is set.
    """

    def __init__(self, file, name, noise=None, seed=0, preprocess=None,
                 function_set=None, extra_data_dir=None,
                 dataset_size_multiplier=None, shuffle_data=True,
                 train_fraction=0.8, experiment_root=None, logdir=None, backup=False):
        output_message = '\n-- Building dataset -----------------\n'
        # Set experiment path
        if experiment_root is None:
            task_root = resource_filename("dsr.task", "regression")
        else:
            task_root = os.path.join(experiment_root)
        # Set data path
        if extra_data_dir is not None:
            data_root = os.path.join(task_root, extra_data_dir)
        else:
            data_root = os.path.join(task_root, "")
        # Load benchmark data if available
        if file is not None:
            benchmark_path = os.path.join(task_root, file)
            benchmark_df = pd.read_csv(benchmark_path, index_col=0, encoding="ISO-8859-1")
            output_message += 'Benchmark path                 : {}\n'.format(benchmark_path)

        # Random number generator used for sampling X values
        seed += zlib.adler32(name.encode("utf-8")) # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
        self.rng = np.random.RandomState(seed)

        self.dataset_size_multiplier = dataset_size_multiplier if dataset_size_multiplier is not None else 1.0

        # Raw dataset
        if "benchmark_df" not in locals() or name not in benchmark_df.index:
            dataset_path = os.path.join(data_root, "{}.csv".format(name))
            data = pd.read_csv(dataset_path, header=None) # Assuming data file does not have header rows
            output_message += 'Loading data from file         : {}\n'.format(dataset_path)
            # Perform some data augmentation if necessary
            if noise is not None and noise != 0:
                print("Warning: Noise will not be applied to real-world dataset.")
            if shuffle_data:
                data = data.sample(frac=1, random_state=self.rng).reset_index(drop=True) # Shuffle rows
            data = data.values
            if preprocess == "standardize_y":
                mean = np.mean(data[:, -1])
                std = np.std(data[:, -1])
                data[:, -1] = (data[:, -1] - mean) / std
            elif preprocess == "standardize_all":
                data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            elif preprocess == "shift_y":
                mean = np.mean(data[:, -1])
                data[:, -1] = data[:, -1] - mean
            elif preprocess == "shift_all":
                data = data - np.mean(data, axis=0)
            elif preprocess == "rescale_y":
                min_ = min(data[:, -1])
                max_ = max(data[:, -1])
                data[:, -1] = (data[:, -1] - min_) / (max_ - min_)
            elif preprocess == "rescale_all":
                min_ = np.min(data, axis=0)
                max_ = np.max(data, axis=0)
                data = (data - min_) / (max_ - min_)
            elif preprocess == None:
                pass
            else:
                assert False, 'Dataset.__init__(): Preprocessing function {} unknown'.format(preprocess)

            self.n_input_var = data.shape[1] - 1

            n_train = int(train_fraction * data.shape[0]) # default: 80-20 train-test split
            assert n_train >= 1, "Dataset.__init__(): Invalid train_fraction: need at least one point in training set."
            if n_train == data.shape[0]:
                print("Warning: Empty test set.")

            self.X_train = data[:n_train, :-1]
            self.y_train = data[:n_train, -1]
            self.X_test = data[n_train:, :-1]
            self.y_test = data[n_train:, -1]

            self.y_train_noiseless = self.y_train.copy()
            self.y_test_noiseless = self.y_test.copy()

            self.sympy_expr = None
            self.numpy_expr = None
            self.fp_constant = None
            self.int_constant = None
            self.train_spec = None
            self.test_spec = None

            if function_set is None:
                function_set_aux = "Real" # Default value of library
            else:
                function_set_aux = function_set

        # Expressions
        else:
            output_message += 'Generating data for benchmark  : {}\n'.format(name)
            row = benchmark_df.loc[name]
            function_set_aux = row["function_set"]
            self.n_input_var = row["variables"]

            # Create symbolic expression
            self.sympy_expr = parse_expr(row["sympy"])
            self.numpy_expr = self.make_numpy_expr(row["numpy"])
            self.fp_constant = "Float" in srepr(self.sympy_expr)
            self.int_constant = "Integer" in srepr(self.sympy_expr)

            # Create X values
            train_spec = ast.literal_eval(row["train_spec"])
            test_spec = ast.literal_eval(row["test_spec"])
            if test_spec is None:
                test_spec = train_spec
            self.X_train = self.make_X(train_spec)
            self.X_test = self.make_X(test_spec)
            self.train_spec = train_spec
            self.test_spec = test_spec

            # Compute y values
            self.y_train = self.numpy_expr(self.X_train)
            self.y_test = self.numpy_expr(self.X_test)
            self.y_train_noiseless = self.y_train.copy()
            self.y_test_noiseless = self.y_test.copy()

            # Add Gaussian noise
            if noise is not None:
                assert noise >= 0, "Dataset.__init__(): Noise must be non-negative."
                y_rms = np.sqrt(np.mean(self.y_train**2))
                scale = noise * y_rms
                self.y_train += self.rng.normal(loc=0, scale=scale, size=self.y_train.shape)
                self.y_test += self.rng.normal(loc=0, scale=scale, size=self.y_test.shape)

        # Load the function set
        if isinstance(function_set_aux, list):
            self.function_set = function_set
        elif isinstance(function_set_aux, str):
            # Create the function set (list of str/float)
            function_set_path = os.path.join(task_root, "function_sets.csv")
            try:
                function_set_df = pd.read_csv(function_set_path, index_col=0)
                self.function_set = function_set_df.loc[function_set_aux].tolist()[0].strip().split(',')
                output_message += 'Loading function set from      : {}\n'.format(function_set_path)
            except:
                assert False, 'Dataset.__init__(): Could not load function set {} from {}'.format(function_set_aux, function_set_path)
            # Replace hard-coded constants with their float values
            def is_number(s):
                try:
                    float(s)
                    return True
                except ValueError:
                    return False
            self.function_set = [float(f) if is_number(f) else f for f in self.function_set]
        else:
            assert False, "Dataset.__init__(): Function set unknown type: {}".format(type(function_set))

        output_message += 'Function set                   : {} --> {}\n'.format(function_set_aux, self.function_set)
        if backup:
            self.save_dataset(logdir, name)
        output_message += '-------------------------------------\n\n'
        print(output_message)

    def make_X(self, spec):
        """Creates X values based on specification"""

        features = []
        for i in range(1, self.n_input_var + 1):

            # Hierarchy: "all" --> "x{}".format(i)
            input_var = "x{}".format(i)
            if "all" in spec:
                input_var = "all"
            elif input_var not in spec:
                input_var = "x1"

            if "U" in spec[input_var]:
                low, high, n = spec[input_var]["U"]
                n = int(n * self.dataset_size_multiplier)
                feature = self.rng.uniform(low=low, high=high, size=n)
            elif "E" in spec[input_var]:
                start, stop, step = spec[input_var]["E"]
                if step > stop - start:
                    n = step
                else:
                    n = int((stop - start)/step) + 1
                n = int(n * self.dataset_size_multiplier)
                feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
            else:
                raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
            features.append(feature)

        # Do multivariable combinations
        if "E" in spec[input_var] and self.n_input_var > 1:
            X = np.array(list(itertools.product(*features)))
        else:
            X = np.column_stack(features)

        return X


    def make_numpy_expr(self, s):

        # This isn't pretty, but unlike sympy's lambdify, this ensures we use
        # our protected functions. Otherwise, some expressions may have large
        # error even if the functional form is correct due to the training set
        # not using protected functions.

        # # Set protected functions
        # for k in function_map.keys():
        #     exec("{} = function_map['{}']".format(k, k))
        # pi = np.pi
        # ln = function_map["log"]

        # Replace function names
        s = s.replace("ln(", "log(")
        s = s.replace("pi", "np.pi")
        s = s.replace("pow", "np.power")
        for k in function_map.keys():
            s = s.replace(k + '(', "function_map['{}'].function(".format(k))

        # Replace variable names
        for i in reversed(range(self.n_input_var)):
            old = "x{}".format(i+1)
            new = "x[:, {}]".format(i)
            s = s.replace(old, new)

        numpy_expr = lambda x : eval(s)

        return numpy_expr


    def pretty(self):
        return pretty(self.sympy_expr)


    def __repr__(self):
        return pretty(self.sympy_expr)

    def save_dataset(self, logdir, name):
        try:
            save_path = os.path.join(logdir,'data_{}.csv'.format(name))
            np.savetxt(
                save_path,
                np.concatenate(
                    (
                        np.hstack((self.X_train, self.y_train[..., np.newaxis])),
                        np.hstack((self.X_test, self.y_test[..., np.newaxis]))
                    ), axis=0),
                delimiter=',', fmt='%1.5f'
            )
            print('Saved dataset to               : {}'.format(save_path))
        except:
            print("Warning: Could not save dataset.")


def plot_dataset(d, output_filename):
    """Plot Dataset with underlying ground truth."""

    # Draw ground truth expression
    bounds = list(list(d.train_spec.values())[0].values())[0][:2]
    x = np.linspace(bounds[0], bounds[1], endpoint=True, num=100)
    y = d.numpy_expr(x[:, None])
    plt.plot(x, y)

    # Draw the actual points
    plt.scatter(d.X_train, d.y_train)

    plt.title(output_filename[:-4], fontsize=7)
    plt.show()


def save_dataset(d, output_filename):
    """Save a Dataset's train and test sets to CSV."""

    regression_data_path = resource_filename("dsr.task", "regression/data/")
    output_filename = os.path.join(regression_data_path, output_filename)
    Xs = [d.X_train, d.X_test]
    ys = [d.y_train, d.y_test]
    output_filenames = [output_filename, output_filename[:-4] + "_test.csv"]
    for X, y, output_filename in zip(Xs, ys, output_filenames):
        print("Saving to {}".format(output_filename))
        y = np.reshape(y, (y.shape[0],1))
        XY = np.concatenate((X,y), axis=1)
        pd.DataFrame(XY).to_csv(output_filename, header=None, index=False)


@click.command()
@click.argument("file", default="benchmarks.csv")
@click.option('--plot', is_flag=True)
@click.option('--save_csv', is_flag=True)
@click.option('--sweep', is_flag=True)
def main(file, plot, save_csv, sweep):
    """Pretty prints and plots all benchmark expressions."""

    regression_path = resource_filename("dsr.task", "regression/")
    benchmark_path = os.path.join(regression_path, file)
    df = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
    names = df["name"].to_list()
    expressions = [parse_expr(expression) for expression in df["sympy"]]
    for expression, name in zip(expressions, names):

        if not name.startswith("Nguyen") and not name.startswith("Constant") and not name.startswith("Custom"):
            continue

        print("{}:\n\n{}\n\n".format(name, indent(pretty(expression), '\t')))
        datasets = []
        output_filenames = []

        # Noiseless
        d = Dataset(file, name, noise=None)
        datasets.append(d)
        output_filename = "{}.csv".format(name)
        output_filenames.append(output_filename)

        # Generate all combinations of noise levels and dataset size multipliers
        if sweep and name.startswith("Nguyen"):
            noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            dataset_size_multipliers = [1.0, 10.0]
            for noise in noises:
                for dataset_size_multiplier in dataset_size_multipliers:
                    d = Dataset(file, name, noise=noise,
                        dataset_size_multiplier=dataset_size_multiplier)
                    datasets.append(d)
                    output_filename = "{}_n{:.2f}_d{:.0f}.csv".format(name, noise, dataset_size_multiplier)
                    output_filenames.append(output_filename)
        # Plot and/or save datasets
        for d, output_filename in zip(datasets, output_filenames):
            if plot and d.X_train.shape[1] == 1:
                plot_dataset(d, output_filename)
            if save_csv:
                save_dataset(d, output_filename)

if __name__ == "__main__":
    main()
