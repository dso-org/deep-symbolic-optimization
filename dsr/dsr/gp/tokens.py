import numpy as np
from dsr.program import Program,  _finish_tokens
from collections import defaultdict
from dsr.subroutines import jit_parents_siblings_at_once

try:
    from deap import gp
    from deap import base
    from deap import tools
    from deap import creator
    from deap import algorithms
except ImportError:
    gp          = None
    base        = None
    tools       = None
    creator     = None
    algorithms  = None

from deap.gp import PrimitiveSet

# Define the name of type for any types. This is a DEAP widget thingy. 
__type__ = object

r"""
    Fast special case version of below. This is mainly used during constraint 
    checking. 
"""
def opt_DEAP_to_math_tokens(individual):

    tokens = np.array([i.name for i in individual], dtype=np.int32)
    
    return tokens

r"""
    This is a base class for accessing DEAP and interfacing it with DSR. 
        
    These are pure symblic components which relate to any symblic task. These are not purely task agnostic and
    are kept seprate from core.
"""
def DEAP_to_math_tokens(individual, tokens_size):

    # Compute unpadded actions
    actions = opt_DEAP_to_math_tokens(individual)

    actions_padded = np.zeros(tokens_size, dtype=np.int32)
    actions_padded[:len(actions)] = actions

    return actions_padded


def math_tokens_to_DEAP(tokens, pset):
    """
    Transforms DSR standard tokens into DEAP format tokens.

    DSR and DEAP format are very similar, but we need to translate it over. 

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    pset : gp.PrimitiveSet

    Returns
    _______
    individual : gp.PrimitiveTree
        This is a specialized list that contains points to element from pset that were mapped based 
        on the translation of the tokens. 
    """
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(tokens, np.ndarray), "Raw tokens are supplied as a numpy array."
    assert isinstance(pset, PrimitiveSet), "You need to supply a valid primitive set for translation."
    assert Program.library is not None, "You have to have an initial program class to supply library token conversions."
    
    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens      = _finish_tokens(tokens)

    plist = [pset.mapping[t] for t in tokens]

    individual = gp.PrimitiveTree(plist)
    
    return individual


def individual_to_dsr_aps(individual, library):
    r"""
        This will convert a deap individual to a DSR action, parent, sibling group.
    """ 

    # Get the action tokens from individuals 
    actions = np.array([t.name for t in individual], dtype=np.int32)

    # Add one dim at the front to be (1 x L)
    actions = np.expand_dims(actions, axis=0) 

    # Get the parent/siblings for 
    parent, sibling     = jit_parents_siblings_at_once(actions, arities=library.arities, parent_adjust=library.parent_adjust)
    
    return actions, parent, sibling
