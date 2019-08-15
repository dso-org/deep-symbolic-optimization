import os
import ast
import itertools
from textwrap import indent

import pandas as pd
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify, pretty, srepr

from dsr.functions import _function_map


class Dataset(object):
    """
    Class used to generate X, y data from a named benchmark expression.

    The numpy expression is used to evaluate the expression using any custom/
    protected functions in _function_map. The sympy expression is only used for
    printing, not function evaluation.

    Parameters
    ----------
    file : str
        Filename of CSV with expressions.

    name : str
        Name of expression.

    seed : int, optional
        Random number seed used to generate data.

    **kwargs : keyword arguments, optional
        Unused. Only included to soak up keyword arguments.
    """

    def __init__(self, file, name, seed=0, **kwargs):

        # Read in benchmark dataset information
        df = pd.read_csv(file, index_col=0, encoding="ISO-8859-1")
        row = df.loc[name]
        self.n_input_var = row["variables"]

        # Create symbolic expression        
        self.sympy_expr = parse_expr(row["sympy"])
        self.numpy_expr = self.make_numpy_expr(row["numpy"])
        self.fp_constant = "Float" in srepr(self.sympy_expr)
        self.int_constant = "Integer" in srepr(self.sympy_expr)        

        # Random number generator used for sampling X values
        self.rng = np.random.RandomState(seed) 

        # Create X values
        train_spec = ast.literal_eval(row["train_spec"])
        test_spec = ast.literal_eval(row["test_spec"])
        self.X_train = self.make_X(train_spec)
        self.X_test = self.make_X(test_spec) if test_spec is not None else self.X_train.copy()

        # Compute y values
        self.y_train = self.numpy_expr(self.X_train)
        self.y_test = self.numpy_expr(self.X_test)

        # Create the function set (list of str)
        function_set_path = os.path.join(os.path.dirname(file), "function_sets.csv")
        df = pd.read_csv(function_set_path, index_col=0)
        self.function_set = df.loc[row["function_set"]].tolist()[0].strip().split(',')

    
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
                feature = self.rng.uniform(low=low, high=high, size=n)
            elif "E" in spec[input_var]:
                start, stop, step = spec[input_var]["E"]
                if step > stop - start:
                    n = step
                else:
                    n = int((stop - start)/step) + 1
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


def __main__():
    """Pretty prints all benchmark expressions."""

    file = "benchmarks.csv"
    exclude_fp_constant = False
    exclude_int_constant = False

    df = pd.read_csv(file, encoding="ISO-8859-1")
    names = df["name"].to_list()
    expressions = [parse_expr(expression) for expression in df["expression"]]
    for expression, name in zip(expressions, names):
        if exclude_fp_constant and "Float" in srepr(expression):
            continue
        if exclude_int_constant and "Integer" in srepr(expression):
            continue
        print("{}:\n\n{}\n\n".format(name, indent(pretty(expression), '\t')))


if __name__ == "__main__":
    main()
