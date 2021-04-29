"""Genetic programming utils."""

import random
import copy
from functools import wraps
from collections import defaultdict

import numpy as np
from deap import gp

from dsr.program import _finish_tokens
from dsr.subroutines import jit_parents_siblings_at_once

__type__ = object


def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        common_types = []
        for idx, node in enumerate(ind2[1:], 1):
            if node.ret in types1 and not node.ret in types2:
                common_types.append(node.ret)
            types2[node.ret].append(idx)
        # common_types = set(types1.keys()).intersection(set(types2.keys()))
        # common_types = [x for x in types1 if x in types2]

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def staticLimit(key, max_value):
    """A fixed version of deap.gp.staticLimit that samples without replacement.
    This prevents returning identical objects, for example if both children of a
    crossover operation are illegal."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # Copy args first in case func mutates them
            keep_inds = [copy.deepcopy(ind) for ind in args]

            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:

                    # Pop a random individual from keep_inds. This ensures we
                    # don't sample the same object twice.
                    pop_index = random.randint(0, len(keep_inds) - 1)
                    new_inds[i] = keep_inds.pop(pop_index)

            return new_inds
        return wrapper
    return decorator


def multi_mutate(individual, expr, pset):
    """Randomly select one of four types of mutation."""

    v = np.random.randint(0, 4)

    if v == 0:
        individual = gp.mutUniform(individual, expr, pset)
    elif v == 1:
        individual = gp.mutNodeReplacement(individual, pset)
    elif v == 2:
        individual = gp.mutInsert(individual, pset)
    elif v == 3:
        individual = gp.mutShrink(individual)

    return individual


def create_primitive_set(lib):
    """Create a DEAP primitive set from a dsr.libraryLibrary."""

    pset = gp.PrimitiveSet("MAIN", len(lib.input_tokens))
    rename_kwargs = {"ARG{}".format(i) : i for i in range(len(lib.input_tokens))}
    for k, v in rename_kwargs.items():

        # pset.renameArguments doesn't actually rename the Primitive.
        # It just affects the pset mapping. So, rename the primitive here.
        pset.mapping[k].name = v

    pset.renameArguments(**rename_kwargs)

    for i, token in enumerate(lib.tokens):
        if token.input_var is None:
            pset.addPrimitive(None, token.arity, name=i)

    return pset


def individual_to_dsr_aps(individual, library):
    """Convert an individual to actions, parents, siblings."""

    actions = np.array([[t.name for t in individual]], dtype=np.int32)
    parent, sibling = jit_parents_siblings_at_once(actions, arities=library.arities, parent_adjust=library.parent_adjust)
    return actions, parent, sibling


def DEAP_to_tokens(individual):
    """
    Convert individual to tokens.

    Parameters
    ----------

    individual : gp.PrimitiveTree
        The DEAP individual.

    Returns
    -------

    tokens : np.array
        The tokens corresponding to the individual.
    """

    tokens = np.array([i.name for i in individual], dtype=np.int32)
    return tokens


def DEAP_to_padded_tokens(individual, max_length):
    """
    Convert individual to tokens padded to max_length.

    Parameters
    ----------

    individual : gp.PrimitiveTree
        The DEAP individual.

    Returns
    -------

    tokens : np.array
        The tokens corresponding to the individual.
    """

    actions = DEAP_to_tokens(individual) # Convert to unpadded actions
    actions_padded = np.zeros(max_length, dtype=np.int32)
    actions_padded[:len(actions)] = actions
    return actions_padded


def tokens_to_DEAP(tokens, pset):
    """
    Convert DSR tokens into DEAP individual.

    Parameters
    ----------
    tokens : np.ndarray
        Tokens corresponding to the individual.

    pset : gp.PrimitiveSet
        Primitive set upon which to build the individual.

    Returns
    _______
    individual : gp.PrimitiveTree
        The DEAP individual.
    """

    tokens = _finish_tokens(tokens)
    plist = [pset.mapping[t] for t in tokens]
    individual = gp.PrimitiveTree(plist)
    return individual


def str_logbook(logbook, header_only=False, startindex=0):
    """
    Pretty print a deap.tools.Logbook. 
    """
    
    if header_only:
        startindex  = 0
        endindex    = 1
    else:
        endindex    = -1
    
    columns = logbook.header
    
    if not columns:
        columns = sorted(logbook[0].keys()) + sorted(logbook.chapters.keys())
                            
    if not logbook.columns_len or len(logbook.columns_len) != len(columns):
        logbook.columns_len = map(len, columns)

    # Start index is set at function call, or is 0 if doing the header

    chapters_txt = {}
    offsets = defaultdict(int)
    for name, chapter in logbook.chapters.items():
        chapters_txt[name] = chapter.__txt__(startindex)
        if startindex == 0:
            offsets[name] = len(chapters_txt[name]) - len(logbook)

    str_matrix = []
    
    for i, line in enumerate(logbook[startindex:endindex]):
        str_line = []
        for j, name in enumerate(columns):
            if name in chapters_txt:
                # Put Chapter over the column label line
                column = chapters_txt[name][i+offsets[name]]
            else:
                # Put the column label
                value = line.get(name, "")
                string = "{0:n}" if isinstance(value, float) else "{0}"
                column = string.format(value)
            logbook.columns_len[j] = max(logbook.columns_len[j], len(column))
            str_line.append(column)
        str_matrix.append(str_line)
                
    if startindex == 0 and logbook.log_header:
        header = []
        nlines = 1
        if len(logbook.chapters) > 0:
            nlines += max(map(len, chapters_txt.values())) - len(logbook) + 1
        header = [[] for i in range(nlines)]
        
        for j, name in enumerate(columns):
            if name in chapters_txt:
                length = max(len(line.expandtabs()) for line in chapters_txt[name])
                blanks = nlines - 2 - offsets[name]
                for i in range(blanks):
                    header[i].append(" " * length)
                header[blanks].append(name.center(length))
                header[blanks+1].append("-" * length)
                for i in range(offsets[name]):
                    header[blanks+2+i].append(chapters_txt[name][i])
            else:
                length = max(len(line[j].expandtabs()) for line in str_matrix)
                for line in header[:-1]:
                    line.append(" " * length)
                header[-1].append(name)
        
        if header_only:
            str_matrix = header  
        else:
            str_matrix = chain(header, str_matrix)
        
    template    = "\t".join("{%i:<%i}" % (i, l) for i, l in enumerate(logbook.columns_len))
    text        = [template.format(*line) for line in str_matrix]
    
    return "\n".join(text)