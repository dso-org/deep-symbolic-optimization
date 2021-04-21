import warnings
from functools import partial
import numpy as np

from dsr.gp import symbolic_math as gp_symbolic_math
from dsr.gp.base import create_primitive_set


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

        
class GPController(gp_symbolic_math.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, config_prior):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        pset = create_primitive_set()
        check_constraint            = gp_symbolic_math.checkConstraint
        hof = tools.HallOfFame(maxsize=1) 
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, config_prior, 
                                           pset, check_constraint, hof)
        




