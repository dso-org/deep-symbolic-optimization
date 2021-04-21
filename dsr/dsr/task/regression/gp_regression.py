import warnings
from functools import partial
import numpy as np

from dsr.gp.base import create_primitive_set
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

        
class GPController(controller_base.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, config_prior):
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, config_prior)
        




