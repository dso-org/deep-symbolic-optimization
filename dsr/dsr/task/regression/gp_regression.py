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
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2)

        elif metric == "rmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2))

        elif metric == "nmse":
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2 / var_y)

        elif metric == "nrmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2 / var_y))
        else:
            raise ValueError("Metric not recognized.")


        return fitness
    
        
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
        pset, const_opt             = self._create_primitive_set(config_training, config_gp_meld, config_task, 
                                                                 n_input_var=dataset.X_train.shape[1], function_set=dataset.function_set)                                         
        eval_func                   = GenericEvaluate(const_opt, dataset, fitness_metric=config_gp_meld["fitness_metric"]) 
        check_constraint            = gp_symbolic_math.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, config_prior, 
                                           pset, eval_func, check_constraint, eval_func.hof)
        




