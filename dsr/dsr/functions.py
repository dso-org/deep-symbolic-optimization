"""Functions used in symbolic regression."""

import numpy as np


class Function(object):
    """
    Callable function class used as nodes in a Program.

    Parameters
    ----------

    function : callable
        Function that returns a np.ndarray with the same shape as its arguments.

    name : str
        Name of the function used for Program visualization.

    arity : int
        Number of arguments.
    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity


    def __call__(self, *args):
        return self.function(*args)


"""Define custom unprotected operators"""
def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))

def expneg(x1):
    return np.exp(-x1)

def n3(x1):
    return np.power(x1, 3)

def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))


# Annotate unprotected ops
unprotected_ops = [
    # Binary operators
    (np.add, "add", 2),
    (np.subtract, "sub", 2),
    (np.multiply, "mul", 2),
    (np.divide, "div", 2),

    # Built-in unary operators
    (np.sin, "sin", 1),
    (np.cos, "cos", 1),
    (np.tan, "tan", 1),
    (np.exp, "exp", 1),
    (np.log, "log", 1),
    (np.sqrt, "sqrt", 1),
    (np.square, "n2", 1),
    (np.negative, "neg", 1),
    (np.abs, "abs", 1),
    (np.maximum, "max", 1),
    (np.minimum, "min", 1),
    (np.tanh, "tanh", 1),
    (np.reciprocal, "inv", 1),

    # Custom unary operators
    (logabs, "logabs", 1),
    (expneg, "expneg", 1),
    (n3, "n3", 1),
    (sigmoid, "sigmoid", 1)
]


"""Define custom protected operators"""
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)

def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def protected_sqrt(x1):
    """Closure of sqrt for negative arguments."""
    return np.sqrt(np.abs(x1))

def protected_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

def protected_expneg(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)

def protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)

def protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)

def protected_sigmoid(x1):
    return 1 / (1 + protected_expneg(x1))


# Annotate protected ops
protected_ops = [
    # Protected binary operators
    (protected_div, "div", 2),

    # Protected unary operators
    (protected_exp, "exp", 1),
    (protected_log, "log", 1),
    (protected_log, "logabs", 1), # Protected logabs is support, but redundant
    (protected_inv, "inv", 1),
    (protected_expneg, "expneg", 1),
    (protected_n2, "n2", 1),
    (protected_n3, "n3", 1),
    (protected_sigmoid, "sigmoid", 1),
]


# Add unprotected ops to function map
function_map = {
    op[1] : Function(*op) for op in unprotected_ops
    }

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op[1]) : Function(*op) for op in protected_ops
    })

UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])
