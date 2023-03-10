"""Genetic programming utils."""

import random
import copy
from functools import wraps
from collections import defaultdict
import time

import numpy as np
from deap import gp

from dso.program import _finish_tokens
from dso.subroutines import jit_parents_siblings_at_once

__type__ = object


class Individual(gp.PrimitiveTree):
    """ Class representing an individual in DEAP's framework. 
        Besides gp.PrimitiveTree, it also contains other information
        related to binding task, such as number and max mutations.
        It can incorporate more information for future tasks. """

    def __init__(self, actions, pset, max_mutations,
                 ind_representation, master_sequence):
        super().__init__(tokens_to_DEAP(actions, pset))

        self.ind_representation = ind_representation
        self.master_sequence = master_sequence
        # work_repr is likely to be in a different representation
        # then it cannot be "linked" to the same 'actions' object
        # otherwise it will mess up with Program.cache
        self.work_repr = actions.copy()

        self.pset = pset
        self.max_mutations = max_mutations
        self.update_num_mutations()

    def __deepcopy__(self, memo):
        """ Override gp.PrimitiveTree's deepcopy. """
        new = Individual(self.tokenized_repr, self.pset,
                         self.max_mutations, self.ind_representation,
                         self.master_sequence)
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        new.update_tree_repr()
        return new

    @property
    def tokenized_repr(self):
        """ Convert to the representation that one can 
            compute rewards from. """
        token_repr = self.work_repr.copy()

        return token_repr
  
    def update_tree_repr(self):
        """ Update gp.PrimitiveTree from the vector representation. """
        self = tokens_to_DEAP(self.tokenized_repr, self.pset)

    def update_num_mutations(self):
        """ Update number of mutations performed wrt master sequence. """
        self.num_mutations = sum(self.work_repr > 0)

    def set_to_zero(self):
        """ Set gp.PrimitiveTree and work_repr to zero. """
        self.num_mutations = 0
        self.work_repr *= 0
        self.update_tree_repr()


# Fix for https://github.com/DEAP/deap/issues/190
# Proposed by https://github.com/EpistasisLab/tpot/pull/412/files
def cxOnePoint(ind1, ind2, **kwargs):
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
            if node.ret in types1 and node.ret not in types2:
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


def cxModifiedPMX(ind1, ind2, **kwargs):
    """Executes a modified two-point crossover on the input :term:`sequence` individuals,
    so that the offsprings respect constraints on the allowed number of mutations.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.

    :returns: A tuple with two new individuals.
    """

    offsp_1 = copy.deepcopy(ind1)
    offsp_2 = copy.deepcopy(ind2)

    # crossover is simpler if offsprings are all zero
    offsp_1.set_to_zero()
    offsp_2.set_to_zero()

    size = min(len(offsp_1.work_repr), len(offsp_2.work_repr))
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # copy parts of parents into offsprings
    offsp_1.work_repr[cxpoint1:cxpoint2] = ind2.work_repr[cxpoint1:cxpoint2].copy()
    offsp_2.work_repr[cxpoint1:cxpoint2] = ind1.work_repr[cxpoint1:cxpoint2].copy()

    # update number of mutations
    offsp_1.update_num_mutations()
    offsp_2.update_num_mutations()

    idx_notsel = [i for i in range(0, cxpoint1)] + \
                    [i for i in range(cxpoint2, size)]

    # fill offspring 1 - from left to right
    for i in idx_notsel:
        if offsp_1.num_mutations < offsp_1.max_mutations: 
            offsp_1.work_repr[i] = int(ind1.work_repr[i])
        else:
            break
        offsp_1.update_num_mutations()

    # fill offspring 2 - from right to left
    for i in idx_notsel[::-1]:
        if offsp_2.num_mutations < offsp_2.max_mutations: 
            offsp_2.work_repr[i] = int(ind2.work_repr[i])
        else:
            break
        offsp_2.update_num_mutations()

    # as work_repr has been modified, we need to update tree
    offsp_1.update_tree_repr()
    offsp_2.update_tree_repr()

    # to enforce fitness computation for these individuals
    del offsp_1.fitness.values, offsp_2.fitness.values

    return offsp_1, offsp_2


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


def multi_mutate(individual, expr, pset, indpb):
    """Randomly select one of four types of mutation."""

    # decide whether or not it will be mutated
    if np.random.rand() <= indpb:
        # choose mutation operator from the list
        v = np.random.randint(0, 4)
        if v == 0:
            individual, = gp.mutUniform(individual, expr, pset)
        elif v == 1:
            individual, = gp.mutNodeReplacement(individual, pset)
        elif v == 2:
            individual, = gp.mutInsert(individual, pset)
        elif v == 3:
            individual, = gp.mutShrink(individual)

    return individual,


def mutConstrainedUniformInt(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from which to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from which to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual.work_repr)
    for i in np.random.permutation(size):
        if random.random() < indpb:
            if individual.num_mutations < individual.max_mutations:
                individual.work_repr[i] = np.random.randint(low, up)
                # update number of mutations
                individual.update_num_mutations()

    return individual,


def mutShuffleIndexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be mutated. Usually this mutation is applied on
    vector of indices.
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be exchanged to
                  another position.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    # notice that mutation is performed at position level
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual.work_repr[i], individual.work_repr[swap_indx] = \
                individual.work_repr[swap_indx], individual.work_repr[i]

    return individual,


def multi_constrained_mutate(individual, expr, pset, indpb):
    """Randomly select one of two types of constrained mutation."""

    # choose mutation operator from the list
    v = np.random.randint(0, 2)
    if v == 0:
        # the sequence is the difference relative to the master sequence
        individual = mutShuffleIndexes(individual, indpb)
    # update tree representation as work_repr might have been modified
    individual[0].update_tree_repr()

    return individual


def rename_token(pset, old_name, new_name):
    """Rename a token. Used mainly to change name back to int with terminals."""

    pset.mapping[new_name] = pset.mapping[old_name]
    pset.mapping[new_name].name = new_name
    pset.mapping[new_name].value = new_name
    del pset.mapping[old_name]
            
    return pset


def create_primitive_set(lib):
    """Create a DEAP primitive set from a dso.libraryLibrary."""

    pset = gp.PrimitiveSet("MAIN", len(lib.input_tokens))
    rename_kwargs = {"ARG{}".format(i): i for i in range(len(lib.input_tokens))}
    for k, v in rename_kwargs.items():

        # pset.renameArguments doesn't actually rename the Primitive.
        # It just affects the pset mapping. So, rename the primitive here.
        pset.mapping[k].name = v

    pset.renameArguments(**rename_kwargs)

    for i, token in enumerate(lib.tokens):
        # Primitives MUST have arity > 0. Deap will error out otherwise. 
        if token.arity > 0:
            pset.addPrimitive(None, token.arity, name=i)
        elif token.function is not None:
            # A zero-arity function, e.g. const or 3.14. This is a terminal, but not an input value like x1.

            # We are forced to use a string. Add a t to make it easier to debug naming. 
            tname = "t{}".format(i)
            # We don't really care about what is in each terminal since they are place holders within deap.
            # So, we set value to None. Name is all we need here since Program will fill in any values for us later.
            pset.addTerminal(None, name=tname)

            # `addTerminal` requires terminal names to be strings. Change back to int. 
            pset = rename_token(pset, tname, i)

    return pset


def individual_to_dso_aps(individual, library):
    """Convert an individual to a trajectory of observations."""

    actions = np.array([[t.name for t in individual]], dtype=np.int32)
    parent, sibling = jit_parents_siblings_at_once(
        actions, arities=library.arities, parent_adjust=library.parent_adjust)
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
    Convert DSO tokens into DEAP individual.

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
