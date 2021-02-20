import random
import operator
import copy
import warnings
from functools import partial, wraps
from operator import attrgetter
import numpy as np

from dsr.functions import function_map, UNARY_TOKENS, BINARY_TOKENS
from dsr.library import Token, PlaceholderConstant
#from dsr.const import make_const_optimizer
#from dsr.program import Program,  _finish_tokens
from dsr.task.regression.dataset import BenchmarkDataset
from dsr.task.regression import regression
from dsr.gp import base as gp_base
from dsr.gp import symbolic_math as gp_symbolic_math
from dsr.gp import const as gp_const
from dsr.gp import tokens as gp_tokens

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


def mse(y, y_hat, var_y):
    return np.mean((y - y_hat)**2)

def rmse(y, y_hat, var_y):
    return np.sqrt(np.mean((y - y_hat)**2))
    
def nmse(y, y_hat, var_y):
    return np.mean((y - y_hat)**2 / var_y)

def nrmse(y, y_hat, var_y):
    return np.sqrt(np.mean((y - y_hat)**2 / var_y))
    
class GenericEvaluate(gp_symbolic_math.GenericEvaluate):
    
    def __init__(self, const_opt, dataset, fitness_metric="nmse", early_stopping=False, threshold=1e-12):
        
        super(GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)

        self.fitness            = self._make_fitness(fitness_metric)
        self.X_train            = dataset.X_train.T
        self.X_test             = dataset.X_test.T
        self.y_train            = dataset.y_train
              
        self.train_fitness      = partial(self.fitness, y=dataset.y_train, var_y=np.var(dataset.y_train))
        self.test_fitness       = partial(self.fitness, y=dataset.y_test,  var_y=np.var(dataset.y_test)) # Function of y_hat

        self.const_opt          = const_opt
        if self.const_opt is not None:
            self.optimize = True
        else:
            self.optimize = False
    
    # This should be replaced by the task provided metric    
    def _make_fitness(self, metric):
        """Generates a fitness function by name"""

        if metric == "mse":
            fitness = mse
        elif metric == "rmse":
            fitness = rmse
        elif metric == "nmse":
            fitness = nmse
        elif metric == "nrmse":
            fitness = nrmse
        else:
            raise ValueError("Metric not recognized.")
            
        '''
        if metric == "mse":
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2)

        elif metric == "rmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2))

        elif metric == "nmse":
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2 / var_y)

        elif metric == "nrmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2 / var_y))

        else:
            raise ValueError("Metric not recognized.")
        '''

        return fitness

    def _single_eval(self, individual, f):
        
        '''
            Notes:
            
            optimizer is in const.py as "scipy" : ScipyMinimize
        
            Sometimes this evaluation can fail. If so, return largest error possible.
        '''
        
        try:
            y_hat   = f(*self.X_train)
        except:
            return np.finfo(np.float).max
        
        y       = self.y_train
        res     = np.mean((y - y_hat)**2)
        
        return res
        
    def __call__(self, individual):

        individual = self._optimize_individual(individual, eval_data_set=self.X_test) # Skips if we are not doing const optimization
    
        return self._finish_eval(individual, self.X_train, self.train_fitness)

        
class GPController(gp_symbolic_math.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        config_dataset              = config_task["dataset"]
        dataset                     = BenchmarkDataset(**config_dataset)
        pset, const_opt             = self._create_primitive_set(dataset, config_training, config_gp_meld)                                         
        eval_func                   = GenericEvaluate(const_opt, dataset, fitness_metric=config_gp_meld["fitness_metric"]) 
        check_constraint            = gp_symbolic_math.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, pset, eval_func, check_constraint, eval_func.hof)
        
        self.get_top_n_programs     = gp_symbolic_math.get_top_n_programs
        self.tokens_to_DEAP         = gp_tokens.math_tokens_to_DEAP
        self.init_const_epoch       = config_gp_meld["init_const_epoch"]
        
    def _create_toolbox(self, pset, eval_func, max_const=None, constrain_const=False, **kwargs):
                
        toolbox, creator    = self._base_create_toolbox(pset, eval_func, parallel_eval=False, **kwargs) 
        const               = "const" in pset.context
        toolbox             = self._create_toolbox_const(toolbox, const, max_const)
        
        return toolbox, creator  

    def _create_primitive_set(self, dataset, config_training, config_gp_meld):
        """Create a DEAP primitive set from DSR functions and consts
        """
        
        assert gp is not None,              "Did not import gp. Is it installed?"
        assert isinstance(dataset, object), "dataset should be a DSR Dataset object" 
        
        const_params                = config_training['const_params']
        max_const                   = config_gp_meld["max_const"]
        
        # Get user constants as well as mutable constants that we optimize (if any)
        user_consts, mutable_consts = gp_const.get_consts()
        
        pset                        = gp_symbolic_math.create_primitive_set(dataset.X_train.shape[1])
        
        # Add primitives
        pset                        = self._add_primitives(pset, function_map, dataset.function_set) 
            
        pset, const_opt             = gp_const.const_opt(pset, mutable_consts, max_const, user_consts, const_params, config_training)
        
        return pset, const_opt
    
    def _call_pre_process(self):
        
        if self.init_const_epoch:
            # Reset all mutable constants when we call DEAP GP?
            self.pset.mapping = gp_const.reset_consts(self.pset.mapping, 1.0)



