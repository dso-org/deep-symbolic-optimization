"""Class for deterministically generating a benchmark dataset from benchmark specifications."""

import os
import ast
import itertools
from pkg_resources import resource_filename
import zlib

import click
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from dsr.functions import function_map


class BenchmarkDataset(object):
    """
    Class used to generate (X, y) data from a named benchmark expression.

    Parameters
    ----------
    name : str
        Name of benchmark expression.

    file : str, optional
        Filename of CSV with benchmark expressions.

    root : str, optional
        Directory containing file and function_sets.csv.

    noise : float, optional
        If not None, Gaussian noise is added to the y values with standard
        deviation = noise * RMS of the noiseless y training values.

    dataset_size_multiplier : float, optional
        Multiplier for size of the dataset.

    seed : int, optional
        Random number seed used to generate data. Checksum on name is added to
        seed.

    logdir : str, optional
        Directory where experiment logfiles are saved.

    backup : bool, optional
        Save generated dataset in logdir if logdir is provided.
    """

    def __init__(self, name, file="benchmarks.csv", root=None, noise=None,
                 dataset_size_multiplier=None, seed=0, logdir=None,
                 backup=False):

        # Load benchmark data
        if root is None:
            root = resource_filename("dsr.task", "regression")
        benchmark_path = os.path.join(root, file)
        benchmark_df = pd.read_csv(benchmark_path, index_col=0, encoding="ISO-8859-1")

        # Set random number generator used for sampling X values
        seed += zlib.adler32(name.encode("utf-8")) # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
        self.rng = np.random.RandomState(seed)

        self.dataset_size_multiplier = dataset_size_multiplier if dataset_size_multiplier is not None else 1.0
        row = benchmark_df.loc[name]
        self.n_input_var = row["variables"]

        # Create symbolic expression
        self.numpy_expr = self.make_numpy_expr(row["expression"])

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

        # Load default function set
        function_set_path = os.path.join(root, "function_sets.csv")
        function_set_df = pd.read_csv(function_set_path, index_col=0)
        function_set_name = row["function_set"]
        self.function_set = function_set_df.loc[function_set_name].tolist()[0].strip().split(',')

        # Backup the dataset
        if backup:
            self.save_dataset(logdir, name)

        output_message = '\n-- Building dataset -----------------\n'
        output_message += 'Benchmark path                 : {}\n'.format(benchmark_path)
        output_message += 'Generated data for benchmark   : {}\n'.format(name)
        output_message += 'Function set path              : {}\n'.format(function_set_path)
        output_message += 'Function set                   : {} --> {}\n'.format(function_set_name, self.function_set)
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
    """Plots all benchmark expressions."""

    regression_path = resource_filename("dsr.task", "regression/")
    benchmark_path = os.path.join(regression_path, file)
    df = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
    names = df["name"].to_list()
    for name in names:

        if not name.startswith("Nguyen") and not name.startswith("Constant") and not name.startswith("Custom"):
            continue

        datasets = []
        output_filenames = []

        # Noiseless
        d = BenchmarkDataset(name=name, file=file, noise=None)
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
