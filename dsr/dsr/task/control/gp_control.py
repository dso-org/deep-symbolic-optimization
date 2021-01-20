import random
import operator
import copy
import warnings
from functools import partial, wraps
from operator import attrgetter
import numpy as np

try:
    import gym
except ImportError:
    gym         = None
    
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

from dsr.functions import function_map, UNARY_TOKENS, BINARY_TOKENS
from dsr.const import make_const_optimizer
from dsr.program import Program,  _finish_tokens, from_str_tokens
from dsr.task.regression.dataset import BenchmarkDataset
from dsr import gp_regression
from dsr import gp_base
from . import utils as U

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
    

# This should be replaced by the task provided metric
def make_fitness(metric):
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

class GenericEvaluate(gp_base.GenericEvaluate):
    
    def __init__(self, const_opt, name, env_kwargs, fitness_metric="nmse",
                 optimize=True, early_stopping=False, threshold=1e-12):
        
        assert gym is not None
        
        super(GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)
        
        assert "Bullet" not in name or pybullet_envs is not None, "Must install pybullet_envs."
        
        if env_kwargs is None:
            env_kwargs = {}
            
        if "Bullet" in name:
            self.env = U.TimeFeatureWrapper(self.env)

        # Define closures for environment and anchor model
        self.env                = gym.make(name, **env_kwargs)
        self.n_actions          = env.action_space.shape[0]
        
        fitness                 = make_fitness(fitness_metric)

        self.const_opt          = const_opt
        self.optimize           = optimize
        
    def _finish_eval(self, individual, X, fitness):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f       = self.toolbox.compile(expr=individual)
            y_hat   = f(*X)
            fit     = (fitness(y_hat=y_hat),)
        
        return fit
    
    def __call__(self, individual):

        assert self.toolbox is not None, "Must set toolbox first."

        if self.optimize:
            # Retrieve symbolic constants
            const_idxs = [i for i, node in enumerate(individual) if node.name == "mutable_const"]

            # HACK: If early stopping threshold has been reached, don't do training optimization
            # Check if best individual has NMSE below threshold on test set
            if self.early_stopping and len(self.hof) > 0 and self._finish_eval(self.hof[0], self.X_test, self.test_fitness)[0] < self.threshold:
                return (1.0,)

        if self.optimize and len(const_idxs) > 0:

            # Objective function for evaluating constants
            def obj(consts):        
                individual = gp_regression._set_const_individuals(const_idxs, consts, individual)        

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f       = self.toolbox.compile(expr=individual)
                    
                    # Sometimes this evaluation can fail. If so, return largest error possible.
                    try:
                        y_hat   = f(*self.X_train)
                    except:
                        return np.finfo(np.float).max
                    
                    y       = self.y_train
                    res     = np.mean((y - y_hat)**2)
                    
                # Sometimes this evaluation can fail. If so, return largest error possible.
                if np.isfinite(res):
                    return res
                else:
                    return np.finfo(np.float).max

            # Do the optimization and set the optimized constants
            x0                  = np.ones(len(const_idxs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimized_consts    = self.const_opt(obj, x0)
            
            individual = self._set_const_individuals(self, const_idxs, optimized_consts, individual) 

        return self._finish_eval(individual, self.X_train, self.train_fitness)
    
            
class GPController(gp_base.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, name,  action_spec, 
                 algorithm=None, anchor=None, env_kwargs=None):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        self.action_spec            = action_spec
        
        self._create_model(self, algorithm, anchor)
        
        pset, const_opt             = self._create_primitive_set(config_task, config_training)                                         
        eval_func                   = GenericEvaluate(const_opt, name, env_kwargs, fitness_metric=config_gp_meld["fitness_metric"]) 
        check_constraint            = gp_regression.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, pset, eval_func, check_constraint, eval_func.hof)
        
        self.get_top_n_programs     = gp_regression.get_top_n_programs
        self.get_top_program        = gp_regression.get_top_program        
        self.tokens_to_DEAP         = gp_regression.tokens_to_DEAP
    
    def _create_model(self, algorithm, anchor, anchor_path=None):
        
        # Load the anchor model (if applicable)
        if "anchor" in self.action_spec:
            # Load custom anchor, if provided, otherwise load default
            if algorithm is not None and anchor is not None:
                U.load_model(algorithm, anchor_path) ### <-----????
            else:
                U.load_default_model(name)
            self.model = U.model
        else:
            self.model = None

    def _create_primitive_set(self, config_task, config_training):
        """Create a DEAP primitive set from DSR functions and consts
        """
        
        assert gp is not None,              "Did not import gp. Is it installed?"
        assert isinstance(dataset, object), "dataset should be a DSR Dataset object" 
        
        symbolic_actions    = []
        action_dim          = None
        for i, spec in enumerate(self.action_spec):
    
            # Action taken from anchor policy
            if spec == "anchor":
                continue
            # Action dimnension being learned
            elif spec is None:
                action_dim = i
            # Pre-specified symbolic policy
            elif isinstance(spec, list) or isinstance(spec, str):
                symbolic_actions[i] = from_str_tokens(spec, optimize=False, skip_cache=True)
        else:
            assert False, "Action specifications must be None, a str/list of tokens, or 'anchor'."
        
        function_set                = config_task['function_set']
        const_params                = config_training['const_params']
        have_const                  = "const" in function_set  
        const_optimizer             = "scipy"
        
        pset                        = gp.PrimitiveSet("MAIN", action_dim)
    
        # Add input variables
        rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(action_dim)}
        pset.renameArguments(**rename_kwargs)
    
        # Add primitives
        pset                    = self._add_primitives(pset, function_map, function_set) 
        pset, const_opt         = gp_regression._const_opt(pset, have_const, const_params)
            
        # Get into Deap Tokens
        self.symbolic_actions   = [self.tokens_to_DEAP(i, pset) for i in symbolic_actions]
            
        return pset, const_opt

    def _create_toolbox(self, pset, eval_func, max_const=None, constrain_const=False, **kwargs):
                
        toolbox, creator    = self._base_create_toolbox(pset, eval_func, **kwargs) 
        const               = "const" in pset.context
        toolbox             = gp_regression._create_toolbox_const(toolbox, const, max_const)
        
        return toolbox, creator      
        
        