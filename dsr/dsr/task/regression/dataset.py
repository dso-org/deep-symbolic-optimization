"""Class for deterministically generating a benchmark dataset from benchmark specifications."""

import os
import ast
import itertools
from pkg_resources import resource_filename
import zlib

import click
import pandas as pd
import numpy as np

from dsr.functions import function_map


class BenchmarkDataset(object):
    """
    Class used to generate (X, y) data from a named benchmark expression.

    Parameters
    ----------
    name : str
        Name of benchmark expression.

    benchmark_source : str, optional
        Filename of CSV describing benchmark expressions.

    root : str, optional
        Directory containing benchmark_source and function_sets.csv.

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

    def __init__(self, name, benchmark_source="benchmarks.csv", root=None, noise=0.0,
                 dataset_size_multiplier=1.0, seed=0, logdir=None,
                 backup=False):
        # Set class variables
        self.name = name
        self.seed = seed
        self.noise = noise if noise is not None else 0.0
        self.dataset_size_multiplier = dataset_size_multiplier if dataset_size_multiplier is not None else 1.0

        # Set random number generator used for sampling X values
        seed += zlib.adler32(name.encode("utf-8")) # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
        self.rng = np.random.RandomState(seed)

        # Load benchmark data
        if root is None:
            root = resource_filename("dsr.task", "regression")
        benchmark_path = os.path.join(root, benchmark_source)
        benchmark_df = pd.read_csv(benchmark_path, index_col=0, encoding="ISO-8859-1")
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
        if self.noise > 0:
            y_rms = np.sqrt(np.mean(self.y_train**2))
            scale = self.noise * y_rms
            self.y_train += self.rng.normal(loc=0, scale=scale, size=self.y_train.shape)
            self.y_test += self.rng.normal(loc=0, scale=scale, size=self.y_test.shape)
        elif self.noise < 0:
            print('WARNING: Ignoring negative noise value: {}'.format(self.noise))

        # Load default function set
        function_set_path = os.path.join(root, "function_sets.csv")
        function_set_df = pd.read_csv(function_set_path, index_col=0)
        function_set_name = row["function_set"]
        self.function_set = function_set_df.loc[function_set_name].tolist()[0].strip().split(',')

        # Prepare status output
        output_message = '\n-- Building dataset -----------------\n'
        output_message += 'Benchmark path                 : {}\n'.format(benchmark_path)
        output_message += 'Generated data for benchmark   : {}\n'.format(name)
        output_message += 'Function set path              : {}\n'.format(function_set_path)
        output_message += 'Function set                   : {} --> {}\n'.format(function_set_name, self.function_set)
        if backup and logdir is not None:
            output_message += self.save(logdir)
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

    def save(self, logdir='./'):
        save_path = os.path.join(logdir,'data_{}_n{:.2f}_d{:.0f}_s{}.csv'.format(
                self.name, self.noise, self.dataset_size_multiplier, self.seed))
        try:
            os.makedirs(logdir, exist_ok=True)
            np.savetxt(
                save_path,
                np.concatenate(
                    (
                        np.hstack((self.X_train, self.y_train[..., np.newaxis])),
                        np.hstack((self.X_test, self.y_test[..., np.newaxis]))
                    ), axis=0),
                delimiter=',', fmt='%1.5f'
            )
            return 'Saved dataset to               : {}\n'.format(save_path)
        except:
            import sys
            e = sys.exc_info()[0]
            print("WARNING: Could not save dataset: {}".format(e))

    def plot(self, logdir='./'):
        """Plot Dataset with underlying ground truth."""
        if self.X_train.shape[1] == 1:
            from matplotlib import pyplot as plt
            save_path = os.path.join(logdir,'plot_{}_n{:.2f}_d{:.0f}_s{}.png'.format(
                    self.name, self.noise, self.dataset_size_multiplier, self.seed))

            # Draw ground truth expression
            bounds = list(list(self.train_spec.values())[0].values())[0][:2]
            x = np.linspace(bounds[0], bounds[1], endpoint=True, num=100)
            y = self.numpy_expr(x[:, None])
            plt.plot(x, y, color='red', linestyle='dashed')
            # Draw the actual points
            plt.scatter(self.X_train, self.y_train)
            # Add a title
            plt.title(
                "{} N:{} M:{} S:{}".format(
                    self.name, self.noise, self.dataset_size_multiplier, self.seed),
                fontsize=7)
            try:
                os.makedirs(logdir, exist_ok=True)
                plt.savefig(save_path)
                print('Saved plot to                  : {}'.format(save_path))
            except:
                import sys
                e = sys.exc_info()[0]
                print("WARNING: Could not plot dataset: {}".format(e))
            plt.close()
        else:
            print("WARNING: Plotting only supported for 2D datasets.")


@click.command()
@click.argument("benchmark_source", default="benchmarks.csv")
@click.option('--plot', is_flag=True)
@click.option('--save_csv', is_flag=True)
@click.option('--sweep', is_flag=True)
def main(benchmark_source, plot, save_csv, sweep):
    """Plots all benchmark expressions."""

    regression_path = resource_filename("dsr.task", "regression/")
    benchmark_path = os.path.join(regression_path, benchmark_source)
    save_dir = os.path.join(regression_path, 'log')
    df = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
    names = df["name"].to_list()
    for name in names:

        if not name.startswith("Nguyen") and not name.startswith("Constant") and not name.startswith("Custom"):
            continue

        datasets = []

        # Noiseless
        d = BenchmarkDataset(
            name=name,
            benchmark_source=benchmark_source)
        datasets.append(d)

        # Generate all combinations of noise levels and dataset size multipliers
        if sweep and name.startswith("Nguyen"):
            noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            dataset_size_multipliers = [1.0, 10.0]
            for noise in noises:
                for dataset_size_multiplier in dataset_size_multipliers:
                    d = BenchmarkDataset(
                        name=name,
                        benchmark_source=benchmark_source,
                        noise=noise,
                        dataset_size_multiplier=dataset_size_multiplier,
                        backup=save_csv,
                        logdir=save_dir)
                    datasets.append(d)

        # Plot and/or save datasets
        for dataset in datasets:
            if plot and dataset.X_train.shape[1] == 1:
                dataset.plot(save_dir)

if __name__ == "__main__":
    main()
