"""Numba-compiled subroutines used for deep symbolic optimization."""

from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def parents_siblings(tokens, arities, parent_adjust):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings of the next element of the sequence.

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.

    Parameters
    __________

    tokens : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    parent_adjust : np.ndarray, dtype=np.int32
        Array of parent sub-library index corresponding to library indices.

    Returns
    _______

    adj_parents : np.ndarray, shape=(N,), dtype=np.int32
        Adjusted parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """
    N, L = tokens.shape

    empty_parent = np.max(parent_adjust) + 1 # Empty token is after all non-empty tokens
    empty_sibling = len(arities) # Empty token is after all non-empty tokens
    adj_parents = np.full(shape=(N,), fill_value=empty_parent, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=empty_sibling, dtype=np.int32)
    # Parallelized loop over action sequences
    for r in prange(N):
        arity = arities[tokens[r, -1]]
        if arity > 0: # Parent is the previous element; no sibling
            adj_parents[r] = parent_adjust[tokens[r, -1]]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-c-1, sibling is the next
                adj_parents[r] = parent_adjust[tokens[r, L - c - 1]]
                siblings[r] = tokens[r, L - c]
                break
    return adj_parents, siblings


@jit(nopython=True, parallel=True)
def ancestors(actions, arities, ancestor_tokens):
    """
    Given a batch of action sequences, determines whether the next element of
    the sequence has an ancestor in ancestor_tokens.

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.

    Parameters
    __________

    actions : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    ancestor_tokens : np.ndarray, dtype=np.int32
        Array of ancestor library indices to check.

    Returns
    _______

    mask : np.ndarray, shape=(N,), dtype=np.bool_
        Mask of whether the next element of each sequence has an ancestor in
        ancestor_tokens.
    """

    N, L = actions.shape
    mask = np.zeros(shape=(N,), dtype=np.bool_)
    # Parallelized loop over action sequences
    for r in prange(N):
        dangling = 0
        threshold = None # If None, current branch does not have trig ancestor
        for c in range(L):
            arity = arities[actions[r, c]]
            dangling += arity - 1
            # Turn "on" if a trig function is found
            # Remain "on" until branch completes
            if threshold is None:
                for trig_token in ancestor_tokens:
                    if actions[r, c] == trig_token:
                        threshold = dangling - 1
                        break
            # Turn "off" once the branch completes
            else:
                if dangling == threshold:
                    threshold = None
        # If the sequences ended "on", then there is a trig ancestor
        if threshold is not None:
            mask[r] = True
    return mask
