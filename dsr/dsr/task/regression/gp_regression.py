import warnings
from functools import partial
import numpy as np

from dsr.task.regression.dataset import BenchmarkDataset
from dsr.gp import symbolic_math as gp_symbolic_math


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
        
class GenericEvaluate(gp_symbolic_math.GenericEvaluate):
    
    def __init__(self, early_stopping=False, threshold=1e-12):
        
        super(GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)


            
    def reward(self, individual, X, fitness):
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f       = self.toolbox.compile(expr=individual)
                y_hat   = f(*X)
                fit     = (fitness(y_hat=y_hat),)
        except:
            fit = (np.finfo(np.float).max,)       
        
        return fit
    
    def __call__(self, individual):

        individual = self._optimize_individual(individual, eval_data_set=self.X_test) # Skips if we are not doing const optimization
    
        return self.reward(individual, self.X_train, self.train_fitness)

        
class GPController(gp_symbolic_math.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, config_prior):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        config_dataset              = config_task["dataset"]
        dataset                     = BenchmarkDataset(**config_dataset)
        pset = self._create_primitive_set()
        eval_func                   = GenericEvaluate() 
        check_constraint            = gp_symbolic_math.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, config_prior, 
                                           pset, eval_func, check_constraint, eval_func.hof)
        




