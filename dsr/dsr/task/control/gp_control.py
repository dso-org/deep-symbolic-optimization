import warnings
import numpy as np
import os
import struct

try:
    import gym
except ImportError:
    gym         = None
    
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

from dsr.task.regression import gp_regression
from dsr.task.control import control
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
        
        # Mix things up a little, not nessesary, but for good measure
        seed_shift  = max(int(1e12), struct.unpack("<L", os.urandom(4))[0]) - int(1e3) 
        
        for i in range(self.n_episodes):
            r_episodes[i]       = control.episode(f, action_dim=self.action_dim, evaluate=False, fix_seeds=True, 
                                                  model=self.model, episode_seed_shift=0,  symbolic_actions=self.symbolic_actions, env=self.env, seed=(i+seed_shift),
                                                  get_action=self._get_action, get_fixed_action=self._get_dsr_action)
                                        
        return r_episodes
    
    def reward(self, individual):
        
        f           = self.toolbox.compile(expr=individual)
        
        # Sometimes this evaluation can fail. If so, return largest error possible.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_episodes = self._gym_loop(individual, f)
            except:
                return [np.finfo(np.float).max]
        
        return [self.optimize_stat(r_episodes) * -1.0]
    
    def __call__(self, individual):

        '''
            NOTE:
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            optimizer is in const.py as "scipy" : ScipyMinimize
        '''
        
        return self.reward(individual)
        
            
class GPController(gp_symbolic_math.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, config_prior):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
    
        name            = config_task["name"]
        action_spec     = config_task["action_spec"]
        n_episodes      = config_task["n_episodes_slice"]
        optimize_stat   = config_task['slice_optimize_stat']
        env_kwargs      = config_task["env_kwargs"] if "env_kwargs" in config_task  else None
        algorithm       = config_task["algorithm"]  if "algorithm" in config_task   else None
        anchor          = config_task["anchor "]    if "anchor " in config_task     else None
        
        self.env, env_kwargs                            = control.make_env(name, env_kwargs)
            
        self.action_spec                                = action_spec
        symbolic_actions, action_dim                    = control.create_symbolic_actions(self.action_spec)
        self.model                                      = control.create_model(action_spec, algorithm, anchor, anchor_path=None)
        
        pset, const_opt                                 = self._create_primitive_set(config_training, config_gp_meld, config_task, 
                                                                                     n_input_var=self.env.observation_space.shape[0])       
        
        eval_func                                       = GenericEvaluate(const_opt, name, self.env, self.model, env_kwargs, 
                                                                          symbolic_actions=symbolic_actions, action_dim=action_dim, 
                                                                          n_episodes=n_episodes, optimize_stat=optimize_stat) 
        
        check_constraint                                = gp_symbolic_math.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, config_prior, 
                                           pset, eval_func, check_constraint, eval_func.hof)
        
   
            
        