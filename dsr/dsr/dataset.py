import ast
from textwrap import indent

import pandas as pd
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify, pretty, srepr


class Dataset(object):
    """
    Class used to generate X, y data from a named benchmark expression.

    Parameters
    ----------
    file : str
        Filename of CSV with expressions.

    name : str
        Name of expression.

    extra_input_var : int, optional
        Number of additional input variables.

    seed : int, optional
        Random number seed used to generate data.

    **kwargs : keyword arguments, optional
        Unused. Only included to soak up keyword arguments.
    """

    def __init__(self, file, name, extra_input_var=0, seed=0, **kwargs):

        # Read in benchmark dataset information
        df = pd.read_csv(file, index_col=0, encoding="ISO-8859-1")
        row = df.loc[name]

        # Create symbolic expression
        self.expression = parse_expr(row["expression"]) # SymPy expression
        self.n_input_var = max(row["variables"], *[int(str(x)[1:]) for x in self.expression.free_symbols]) + extra_input_var
        self.input_vars = symbols(' '.join('x{}'.format(i+1) for i in range(self.n_input_var))) # Symbols for input variables
        self.f = lambdify(self.input_vars, self.expression, modules=np) # Vectorized lambda function

        # Characterize the expression
        self.fp_constant = "Float" in srepr(self.expression)
        self.int_constant = "Integer" in srepr(self.expression)

        # Random number generator used for sampling X values
        self.rng = np.random.RandomState(seed) 

        # Create X values
        train_spec = ast.literal_eval(row["train_spec"])
        test_spec = ast.literal_eval(row["test_spec"])
        self.X_train = self.make_X(train_spec)
        self.X_test = self.make_X(test_spec) if test_spec is not None else self.X_train.copy()

        # Compute y values
        self.y_train = self.f(*self.X_train.T)
        self.y_test = self.f(*self.X_test.T)

    
    def make_X(self, spec):
        """Creates X values based on specification"""

        features = []
        for i in range(1, self.n_input_var + 1):

            # Hierarchy: "all" --> "x{}".format(i) --> "x1" (i.e. when extra_input_var > 0)
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
                n = int((stop - start)/step)
                feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
            else:
                raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
            
            features.append(feature)

        X = np.column_stack(features)
        return X


    def pretty(self):
        return pretty(self.expression)


    def __repr__(self):
        return pretty(self.expression)


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
