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
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify, pretty, srepr

from dsr.functions import _function_map


class Dataset(object):
    """
    Class used to generate X, y data from a named benchmark expression or raw
    dataset.

    The numpy expression is used to evaluate the expression using any custom/
    protected functions in _function_map. The sympy expression is only used for
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

    function_lib : list, optional
        List of operators to use in library. If None, use default. Otherwise,
        this will override any default, e.g. when using a benchmark expression.

    extra_data_dir : str, optional
        Absolute path of directory to look for dataset files. If None, uses
        default path dsr/data.

    dataset_size_multiplier : float, optional
        Multiplier for size of the dataset. Only works for expressions.

    **kwargs : keyword arguments, optional
        Unused. Only included to soak up keyword arguments.
    """

    def __init__(self, file, name, noise=None, seed=0, preprocess=None,
                 function_lib=None, extra_data_dir=None, 
                 dataset_size_multiplier=None, **kwargs):

        # Read in benchmark dataset information
        task_root = resource_filename("dsr.task", "regression")
        root = os.path.join(task_root, "data") # Root data directory
        benchmark_path = os.path.join(task_root, file)
        df = pd.read_csv(benchmark_path, index_col=0, encoding="ISO-8859-1")

        # Random number generator used for sampling X values
        seed += zlib.adler32(name.encode("utf-8")) # Different seed for each name, otherwise two benchmarks with the same domain will always have the same X values
        self.rng = np.random.RandomState(seed)

        self.dataset_size_multiplier = dataset_size_multiplier if dataset_size_multiplier is not None else 1.0

        # Load raw dataset from external directory
        root_changed = False
        if extra_data_dir is not None:
            root = os.path.join(extra_data_dir,"")
            root_changed = True

        # Raw dataset
        if name not in df.index:

            if noise is not None and noise != 0:
                print("Warning: Noise will not be applied to real-world dataset.")

            dataset_path = os.path.join(root, name + ".csv")
            data = pd.read_csv(dataset_path)
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
            
            self.n_input_var = data.shape[1] - 1

            n_train = int(0.8 * data.shape[0]) # 80-20 train-test split

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

            function_set = "Real" # Default value of library

        # Expression dataset
        else:
            row = df.loc[name]
            function_set = row["function_set"]
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
                assert noise >= 0, "Noise must be non-negative."
                y_rms = np.sqrt(np.mean(self.y_train**2))
                scale = noise * y_rms
                self.y_train += self.rng.normal(loc=0, scale=scale, size=self.y_train.shape)
                self.y_test += self.rng.normal(loc=0, scale=scale, size=self.y_test.shape)

        # If root has changed
        if root_changed:
            root = resource_filename("dsr.task", "regression/data/")
            
        # Create the function set (list of str)
        function_set_path = os.path.join(task_root, "function_sets.csv")
        df = pd.read_csv(function_set_path, index_col=0)
        self.function_set = df.loc[function_set].tolist()[0].strip().split(',')

        # Overwrite the function set
        if function_lib is not None:
            self.function_set = function_lib
    
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
        # for k in _function_map.keys():
        #     exec("{} = _function_map['{}']".format(k, k))
        # pi = np.pi
        # ln = _function_map["log"]

        # Replace function names
        s = s.replace("ln(", "log(")
        s = s.replace("pi", "np.pi")
        s = s.replace("pow", "np.power")
        for k in _function_map.keys():
            s = s.replace(k + '(', "_function_map['{}'].function(".format(k))

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


@click.command()
@click.argument("file", default="benchmarks.csv")
@click.option("--noise", default=None, type=float)
def main(file, noise):
    """Pretty prints and plots all benchmark expressions."""

    from matplotlib import pyplot as plt

    data_path = resource_filename("dsr.task", "regression/data/")
    benchmark_path = os.path.join(data_path, file)
    df = pd.read_csv(benchmark_path, encoding="ISO-8859-1")
    names = df["name"].to_list()
    expressions = [parse_expr(expression) for expression in df["sympy"]]
    for expression, name in zip(expressions, names):
        print("{}:\n\n{}\n\n".format(name, indent(pretty(expression), '\t')))

        if "Nguyen" not in name:
            continue

        d = Dataset(file, name, noise=noise)
        if d.X_train.shape[1] == 1:

            # Draw ground truth expression
            bounds = list(list(d.train_spec.values())[0].values())[0][:2]
            x = np.linspace(bounds[0], bounds[1], endpoint=True, num=100)
            y = d.numpy_expr(x[:, None])
            plt.plot(x, y)

            # Draw the actual points
            plt.scatter(d.X_train, d.y_train)
            plt.show()


if __name__ == "__main__":
    main()
