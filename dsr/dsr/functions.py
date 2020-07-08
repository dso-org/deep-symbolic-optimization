"""Protected functions used in symbolic regression."""

import numpy as np
from gplearn.functions import _function_map, _Function

# Temporary hack
Function = _Function
function_map = _function_map


def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)


def _proteceted_exponent_negative(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)

def _protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)


def _protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)


def _protected_division_ignore_overflow(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


exp = Function(function=_protected_exponent, name='exp', arity=1)
expneg = Function(function=_proteceted_exponent_negative, name='expneg', arity=1)
n2 = Function(function=_protected_n2, name='n2', arity=1)
n3 = Function(function=_protected_n3, name='n3', arity=1)
tanh = Function(function=np.tanh, name='tanh', arity=1)
div = Function(function=_protected_division_ignore_overflow, name='div', arity=2)

function_map.update({
    'exp' : exp,
    'expneg' : expneg,
    'n2' : n2,
    'n3' : n3,
    'tanh' : tanh,
    'div' : div
    })
