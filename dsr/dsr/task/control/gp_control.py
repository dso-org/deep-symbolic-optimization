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
    



class GenericEvaluate(gp_regression.GenericEvaluate):
    
    def __init__(self, const_opt, name, env, model, env_kwargs, symbolic_actions=None, action_dim=None, n_episodes=5, 
                 early_stopping=False, threshold=1e-12):
        
        assert gym is not None
        
        super(gp_regression.GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)
        
        # Define closures for environment and anchor model
        
        #self.env                = gym.make(name, **env_kwargs)
        self.env                = env
        
        self.fitness            = None

        self.const_opt          = const_opt
        self.name               = name
        self.model              = model
        self.symbolic_actions   = symbolic_actions
        self.action_dim         = action_dim
        self.n_episodes         = n_episodes
        self.threshold          = threshold
                
        if self.const_opt is not None:
            self.optimize = True
        else:
            self.optimize = False
        
        self.early_stopping     = False # Not supported since it would have to call gym twice to check it. 
            
    def _get_dsr_action(self, p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""

        action = p.execute(np.array([obs]))[0]

        return action
    
    def _gym_loop(self, individual, f):
        
        r_episodes = np.zeros(self.n_episodes, dtype=np.float64) # Episodic rewards for each episode
        
        for i in range(self.n_episodes):
        
            self.env.seed(i)
            obs = self.env.reset()
            
            done = False
            while not done:
        
                #print("*****************************************")
                if self.model is not None:
                    action, _   = self.model.predict(obs)
                else:
                    action      = np.zeros(self.env.action_space.shape, dtype=np.float32)
                
                #print("********")
                #print(action)    
                
                for j, fixed_p in self.symbolic_actions.items():
                    action[j]   = self._get_dsr_action(fixed_p, obs)
                
                #print("********")
                #print(action)
                
                if self.action_dim is not None:
                    action[self.action_dim] = f(*obs)
                
                #print("********")
                #print(action)
                
                action[np.isnan(action)]    = 0.0 # Replace NaNs with zero
                action                      = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                #print("********")
                #print(action)
                
                obs, r, done, _             = self.env.step(action) # Does r get small as we get better?
                r_episodes[i] += r
                                
        return r_episodes
    
    def _single_eval(self, individual, f):
        
        #r_episodes = self._gym_loop(individual, f)
        
        # Sometimes this evaluation can fail. If so, return largest error possible.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_episodes = self._gym_loop(individual, f)
            except:
                return [np.finfo(np.float).max]
              
        return [np.mean(r_episodes) * -1.0]
    
    def _finish_eval(self, individual, f):
        
        raise NotImplementedError 
    
    def __call__(self, individual):

        '''
            NOTE:
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            optimizer is in const.py as "scipy" : ScipyMinimize
        '''
        
        individual  = self._optimize_individual(individual) # Skips if we are not doing const optimization
        f           = self.toolbox.compile(expr=individual)
        ret         = self._single_eval(individual, f)
        #print("Return {} val {}".format(individual,ret))
        return ret
        
    
            
class GPController(gp_base.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training):
    
        name            = config_task["name"]
        action_spec     = config_task["action_spec"]
        n_episodes      = config_task["n_episodes_train"]
        env_kwargs      = config_task["env_kwargs"] if "env_kwargs" in config_task  else None
        algorithm       = config_task["algorithm"]  if "algorithm" in config_task   else None
        anchor          = config_task["anchor "]    if "anchor " in config_task     else None
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        assert "Bullet" not in name or pybullet_envs is not None, "Must install pybullet_envs."
        
        if env_kwargs is None:
            env_kwargs = {}
        
        self.env                                        = gym.make(name, **env_kwargs)
        
        if "Bullet" in name:
            self.env = U.TimeFeatureWrapper(self.env)
            
        self.action_spec                                = action_spec
        
        self._create_model(self, algorithm, anchor)
        
        pset, const_opt, symbolic_actions, action_dim   = self._create_primitive_set(config_task, config_training, config_gp_meld)                                         
        eval_func                                       = GenericEvaluate(const_opt, name, self.env, self.model, env_kwargs, symbolic_actions=symbolic_actions, action_dim=action_dim, 
                                                              n_episodes=n_episodes) 
        check_constraint                                = gp_regression.checkConstraint
        
        super(GPController, self).__init__(config_gp_meld, config_task, config_training, pset, eval_func, check_constraint, eval_func.hof)
        
        self.get_top_n_programs                         = gp_regression.get_top_n_programs
        self.get_top_program                            = gp_regression.get_top_program        
        self.tokens_to_DEAP                             = gp_regression.tokens_to_DEAP
    
    def _create_model(self, algorithm, anchor, anchor_path=None):
        
        # Load the anchor model (if applicable)
        if "anchor" in self.action_spec:
            # Load custom anchor, if provided, otherwise load default
            if algorithm is not None and anchor is not None and anchor_path is not None:
                U.load_model(algorithm, anchor_path)
            else:
                U.load_default_model(name)
            self.model = U.model
        else:
            self.model = None

    def _create_primitive_set(self, config_task, config_training, config_gp_meld):
        """Create a DEAP primitive set from DSR functions and consts
        """
        
        assert gp is not None,              "Did not import gp. Is it installed?"
        
        symbolic_actions    = {}
        action_dim          = None
        n_input_var         = self.env.observation_space.shape[0]
        
        for i, spec in enumerate(self.action_spec):
    
            # Action taken from anchor policy
            if spec == "anchor":
                continue
            
            # Action dimnension being learned
            if spec is None:
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
        max_const                   = config_gp_meld["max_const"]
        
        # Get user constants as well as mutable constants that we optimize (if any)
        user_consts                 = [t for i, t in enumerate(Program.library.tokens) if t.arity == 0 and t.input_var is None and t.name != "const"] 
        mutable_consts              = len([t for i, t in enumerate(Program.library.tokens) if t.name == "const"])
        
        pset                        = gp.PrimitiveSet("MAIN", n_input_var)
    
        # Add input variables
        rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(n_input_var)}
        pset.renameArguments(**rename_kwargs)
    
        # Add primitives
        pset                    = self._add_primitives(pset, function_map, function_set) 
        pset, const_opt         = gp_regression._const_opt(pset, mutable_consts, max_const, user_consts, const_params, config_training)
            
        # Get into Deap Tokens
        #self.symbolic_actions   = [self.tokens_to_DEAP(i, pset) for i in symbolic_actions]
        self.symbolic_actions   = symbolic_actions
            
        return pset, const_opt, symbolic_actions, action_dim

    def _create_toolbox(self, pset, eval_func, max_const=None, constrain_const=False, **kwargs):
                
        toolbox, creator    = self._base_create_toolbox(pset, eval_func, **kwargs) 
        const               = "const" in pset.context
        toolbox             = gp_regression._create_toolbox_const(toolbox, const, max_const)
        
        return toolbox, creator      
        
        