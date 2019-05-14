from itertools import chain
import collections

MAX_SEQUENCE_LENGTH = 15
n_inputs = 2

# Library of operators/symbols
binary = ["Add", "Mul"] # binary operators
unary = ["sin", "cos"] # unary operators
leaf = ["x{}".format(i+1) for i in range(n_inputs)] # features = input variabels = leaf nodes
library = binary + unary + leaf
n_children = [2 for _ in binary] + [1 for _ in unary] + [0 for _ in leaf]
choices = list(range(len(library)))
n_choices = len(choices)

# Converts an int traversal to str traversal and vice versa
def convert(traversal):
    assert type(traversal) == list
    assert len(traversal) > 0
    if type(traversal[0]) == str:
        return [library.index(s) for s in traversal]
    return [library[i] for i in traversal]

# Recurisvely flatten a list
def _flatten(L):
    if isinstance(L, list):
        for item in L:
            yield from _flatten(item)
    else:
        yield L
def flatten(L):
    return list(_flatten(L))