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
from dsr.task.regression import gp_regression
from dsr.task.control import control
from dsr.gp import base as gp_base
from dsr.gp import symbolic_math as gp_symbolic_math
from dsr.gp import const as gp_const
from dsr.gp import tokens as gp_tokens
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
    
class GenericEvaluate(gp_regression.GenericEvaluate):
    
    def __init__(self, const_opt, name, env, model, env_kwargs, symbolic_actions=None, action_dim=None, n_episodes=5, 
                 early_stopping=False, optimize_stat="min", threshold=1e-12):
        
        assert gym is not None
        
        super(gp_regression.GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)
             
        self.env                = env
        
        self.fitness            = None

        self.const_opt          = const_opt
        self.name               = name
        self.model              = model
        self.symbolic_actions   = symbolic_actions
        self.action_dim         = action_dim
        self.n_episodes         = n_episodes
        self.optimize_stat      = optimize_stat
        self.threshold          = threshold
                
        if self.const_opt is not None:
            self.optimize = True
        else:
            self.optimize = False
            
        if optimize_stat == "mean":
            self.optimize_stat      = np.mean
        elif optimize_stat == "median":
            self.optimize_stat      = np.median
        elif optimize_stat == "max":
            # Return the best sample of the n 
            self.optimize_stat      = np.amax
        elif optimize_stat == "min":
            # Return the worst sample of the n
            self.optimize_stat      = np.amin
        else:
            print("Got unknown optimize_stat \"{}\"".format(self.optimize_stat))
            raise NotImplementedError
        
        self.early_stopping     = False # Not supported since it would have to call gym twice to check it. 
            
    def _get_dsr_action(self, p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""

        return p.execute(np.array([obs]))[0]
    
    def _get_action(self, f, obs):
        
        return f(*obs)
    
    def _gym_loop(self, individual, f):
        
        r_episodes = np.zeros(self.n_episodes, dtype=np.float64) # Episodic rewards for each episode
        
        for i in range(self.n_episodes):
            r_episodes[i]       = control.episode(f, action_dim=self.action_dim, evaluate=False, fix_seeds=True, 
                                                  model=self.model, episode_seed_shift=0,  symbolic_actions=self.symbolic_actions, env=self.env, seed=i,
                                                  get_action=self._get_action, get_fixed_action=self._get_dsr_action)
                                        
        return r_episodes
    
    def _single_eval(self, individual, f):
        
        # Sometimes this evaluation can fail. If so, return largest error possible.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_episodes = self._gym_loop(individual, f)
            except:
                return [np.finfo(np.float).max]
        
        return [self.optimize_stat(r_episodes) * -1.0]

    def _finish_eval(self, individual, f):
        
        raise NotImplementedError 
    
    def __call__(self, individual):

        '''
            NOTE:
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            optimizer is in const.py as "scipy" : ScipyMinimize
        '''
        
        #individual  = self._optimize_individual(individual, eval_dataset=None) # Skips if we are not doing const optimization
        f           = self.toolbox.compile(expr=individual)
        ret         = self._single_eval(individual, f)
        
        return ret
        
            
class GPController(gp_symbolic_math.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training):
    
        name            = config_task["name"]
        action_spec     = config_task["action_spec"]
        n_episodes      = config_task["n_episodes_slice"]
        optimize_stat   = config_task['slice_optimize_stat']
        env_kwargs      = config_task["env_kwargs"] if "env_kwargs" in config_task  else None
        algorithm       = config_task["algorithm"]  if "algorithm" in config_task   else None
        anchor          = config_task["anchor "]    if "anchor " in config_task     else None
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
   
        self.env, env_kwargs                            = control.make_env(name, env_kwargs)
            
        self.action_spec                                = action_spec
        
        self.model                                      = control.create_model(action_spec, algorithm, anchor, anchor_path=None)
        
        pset, const_opt, symbolic_actions, action_dim   = self._create_primitive_set(config_task, config_training, config_gp_meld)                                         
        eval_func                                       = GenericEvaluate(const_opt, name, self.env, self.model, env_kwargs, symbolic_actions=symbolic_actions, action_dim=action_dim, 
                                                              n_episodes=n_episodes, optimize_stat=optimize_stat) 
        check_constraint                                = gp_symbolic_math.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, pset, eval_func, check_constraint, eval_func.hof)
        
        self.get_top_n_programs                         = gp_symbolic_math.get_top_n_programs     
        self.tokens_to_DEAP                             = gp_tokens.math_tokens_to_DEAP
   
    def _create_toolbox(self, pset, eval_func, max_const=None, constrain_const=False, **kwargs):
                
        toolbox, creator    = self._base_create_toolbox(pset, eval_func, parallel_eval=True, **kwargs) 
        const               = "const" in pset.context
        toolbox             = self._create_toolbox_const(toolbox, const, max_const)
        
        return toolbox, creator 
   
    def _create_primitive_set(self, config_task, config_training, config_gp_meld):
        """Create a DEAP primitive set from DSR functions and consts
        """
        
        assert gp is not None,              "Did not import gp. Is it installed?"
        
        n_input_var                 = self.env.observation_space.shape[0]
                
        symbolic_actions, action_dim    = control.create_symbolic_actions(self.action_spec)
        
        function_set                = config_task['function_set']
        const_params                = config_training['const_params']
        have_const                  = "const" in function_set  
        const_optimizer             = "scipy"
        max_const                   = config_gp_meld["max_const"]
        
        # Get user constants as well as mutable constants that we optimize (if any)
        user_consts, mutable_consts = gp_const.get_consts()
        
        pset                        = gp_symbolic_math.create_primitive_set(n_input_var)

        # Add primitives
        pset                        = self._add_primitives(pset, function_map, function_set) 
        pset, const_opt             = gp_const.const_opt(pset, mutable_consts, max_const, user_consts, const_params, config_training)
            
        # Get into Deap Tokens
        #self.symbolic_actions   = [self.tokens_to_DEAP(i, pset) for i in symbolic_actions]
        self.symbolic_actions   = symbolic_actions
            
        return pset, const_opt, symbolic_actions, action_dim

   
            
        