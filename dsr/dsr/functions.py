"""Functions used in symbolic regression."""

import numpy as np
from fractions import Fraction

GAMMA = 0.57721566490153286060651209008240243104215933593992


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

    complexity : int
        Complexity of the operator.
    """

    def __init__(self, function, name, arity, complexity):
        self.function = function
        self.name = name
        self.arity = arity
        self.complexity = complexity


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

def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val)+1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)


# Annotate unprotected ops
unprotected_ops = [
    # Binary operators
    Function(np.add, "add", arity=2, complexity=1),
    Function(np.subtract, "sub", arity=2, complexity=1),
    Function(np.multiply, "mul", arity=2, complexity=1),
    Function(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    Function(np.sin, "sin", arity=1, complexity=3),
    Function(np.cos, "cos", arity=1, complexity=3),
    Function(np.tan, "tan", arity=1, complexity=4),
    Function(np.exp, "exp", arity=1, complexity=4),
    Function(np.log, "log", arity=1, complexity=4),
    Function(np.sqrt, "sqrt", arity=1, complexity=4),
    Function(np.square, "n2", arity=1, complexity=2),
    Function(np.negative, "neg", arity=1, complexity=1),
    Function(np.abs, "abs", arity=1, complexity=2),
    Function(np.maximum, "max", arity=1, complexity=4),
    Function(np.minimum, "min", arity=1, complexity=4),
    Function(np.tanh, "tanh", arity=1, complexity=4),
    Function(np.reciprocal, "inv", arity=1, complexity=2),

    # Custom unary operators
    Function(logabs, "logabs", arity=1, complexity=4),
    Function(expneg, "expneg", arity=1, complexity=4),
    Function(n3, "n3", arity=1, complexity=3),
    Function(sigmoid, "sigmoid", arity=1, complexity=4),
    Function(harmonic, "harmonic", arity=1, complexity=4)
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
    Function(protected_div, "div", arity=2, complexity=2),

    # Protected unary operators
    Function(protected_exp, "exp", arity=1, complexity=4),
    Function(protected_log, "log", arity=1, complexity=4),
    Function(protected_log, "logabs", arity=1, complexity=4), # Protected logabs is support, but redundant
    Function(protected_sqrt, "sqrt", arity=1, complexity=4),
    Function(protected_inv, "inv", arity=1, complexity=2),
    Function(protected_expneg, "expneg", arity=1, complexity=4),
    Function(protected_n2, "n2", arity=1, complexity=2),
    Function(protected_n3, "n3", arity=1, complexity=3),
    Function(protected_sigmoid, "sigmoid", arity=1, complexity=4),
]

# Add unprotected ops to function map
function_map = {
    op.name : op for op in unprotected_ops
    }

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op.name) : op for op in protected_ops
    })

UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])
