import numpy as np
from functools import partial, wraps
import copy
import random
import operator
import warnings

from dsr.functions import function_map
from dsr.gp import tokens as gp_tokens
from dsr.gp import const as gp_const
from dsr.gp import controller_base

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



# This is called when we mate or mutate and individual
def checkConstraint(max_length, min_length, max_depth, joint_prior_violation):
    """Check a varety of constraints on a memeber. These include:
        Max Length, Min Length, Max Depth, Trig Ancestors and inversion repetse. 
        
        This is a decorator function that attaches to mutate or mate functions in
        DEAP.
                
        >>> This has been tested and gives the same answer as the old constraint 
            function given trig and inverse constraints. 
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds   = [copy.deepcopy(ind) for ind in args]      # The input individual(s) before the wrapped function is called 
            new_inds    = list(func(*args, **kwargs))               # Calls the wrapped function and returns results
                        
            for i, ind in enumerate(new_inds):
                
                l = len(ind)
                
                if l > max_length:
                    new_inds[i] = random.choice(keep_inds)
                elif l < min_length:
                    new_inds[i] = random.choice(keep_inds)
                elif operator.attrgetter("height")(ind) > max_depth:
                    new_inds[i] = random.choice(keep_inds)
                else:  
                    if joint_prior_violation(new_inds[i]):
                        new_inds[i] = random.choice(keep_inds)                    
            return new_inds

        return wrapper

    return decorator


# This is called when we randomly generate a new individual
def popConstraint(joint_prior_violation):
    r""" Check a varety of constraints on a member. This function can optionally run 
         each time we generate a new individual from scratch. 
        
         This is a decorator function that attaches to the individual function in
         DEAP.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            while(True):
                inds    = func(*args, **kwargs)               # Calls the wrapped function and returns results
                
                if joint_prior_violation(inds):
                    continue
                else:
                    break
                                        
            return inds

        return wrapper

    return decorator  


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    #prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_),
        'inv': lambda *args_: "Pow(-1)".format(*args_),
        'neg': lambda *args_: "Mul(-1)".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


class GPController(controller_base.GPController):
    
    def __init__(self, config_gp_meld, *args, **kwargs):
        
        super(GPController, self).__init__(config_gp_meld, *args, **kwargs)
        
        # Get a mapping to the conversion functions used to get Deap to tokens and back
        # These are ones used in symbolic math. 
        self.tokens_to_DEAP                             = gp_tokens.math_tokens_to_DEAP
        self.DEAP_to_tokens                             = gp_tokens.DEAP_to_math_tokens
        self.init_const_epoch                           = config_gp_meld["init_const_epoch"]
            
    def _create_toolbox(self, pset, max_const=None, constrain_const=False, parallel_eval=False, **kwargs):
        
        # Call the base class toolbox creator then do some special case things needed for symbolic math
        toolbox, creator    = self._base_create_toolbox(pset, parallel_eval=parallel_eval, **kwargs) 
        const               = "const" in pset.context
        toolbox             = self._create_toolbox_const(toolbox, const, max_const, constrain_const)
        
        return toolbox, creator  
    
    def _create_toolbox_const(self, toolbox, const, max_const, constrain_const):
     
        # If we have constants and a defined maximum number, put the constraint in here               
        if const and max_const is not None:
            assert isinstance(max_const,int)
            assert max_const >= 0
            num_const = lambda ind : len([node for node in ind if node.name.startwith("mutable_const_")])
            toolbox.decorate("mate",        gp.staticLimit(key=num_const, max_value=max_const))
            toolbox.decorate("mutate",      gp.staticLimit(key=num_const, max_value=max_const))
    
        if const and constrain_const is True:
            toolbox.decorate("mate",        gp.staticLimit(key=self.check_constraint, max_value=0))
            toolbox.decorate("mutate",      gp.staticLimit(key=self.check_constraint, max_value=0))
        
        return toolbox 


    def _call_pre_process(self):

        pass
