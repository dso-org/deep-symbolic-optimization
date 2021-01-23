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
    



class GenericEvaluate(gp_regression.GenericEvaluate):
    
    def __init__(self, const_opt, name, env_kwargs, symbolic_actions=None, action_dim=None, n_episodes=5, threshold=1e-12):
        
        assert gym is not None
        
        super(gp_base.GenericEvaluate, self).__init__(early_stopping=early_stopping, threshold=threshold)
        
        assert "Bullet" not in name or pybullet_envs is not None, "Must install pybullet_envs."
        
        if env_kwargs is None:
            env_kwargs = {}
            
        if "Bullet" in name:
            self.env = U.TimeFeatureWrapper(self.env)

        # Define closures for environment and anchor model
        self.env                = gym.make(name, **env_kwargs)
        self.n_actions          = env.action_space.shape[0]
        
        self.fitness            = None

        self.const_opt          = const_opt
        self.name               = name
        self.symbolic_action    = symbolic_actions
        self.action_dim         = action_dim
        self.n_episodes         = n_episodes
        self.threshold          = threshold
        
        if self.const_opt is not None:
            self.optimize = True
        else:
            self.optimize = False
        
        self.early_stopping     = False # Not supported since it would have to call gym twice to check it. 
            
    def _get_dsr_action(p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""

        action = p.execute(np.array([obs]))[0]

        return action
    
    def _gym_loop(self, individual, f):
        
        r_episodes = np.zeros(self.n_episodes_train, dtype=np.float64) # Episodic rewards for each episode
        
        for i in range(n_episodes):
        
            self.env.seed(i)
            obs = self.env.reset()
            
            done = False
            while not done:
        
                if self.model is not None:
                    action, _   = self.model.predict(obs)
                else:
                    action      = np.zeros(self.env.action_space.shape, dtype=np.float32)
                    
                for j, fixed_p in self.symbolic_actions.items():
                    action[j]   = self._get_dsr_action(self.fixed_p, obs)
                    
                if self.action_dim is not None:
                    action[self.action_dim] = f(*obs)
                    
                action[np.isnan(action)]    = 0.0 # Replace NaNs with zero
                action                      = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                obs, r, done, _             = self.env.step(action) # Does r get small as we get better?
                r_episodes[i] += r
                                
        return r_episodes
    
    def _const_opt_eval(self, individual, f):
        
        # Sometimes this evaluation can fail. If so, return largest error possible.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r_episodes = self._gym_loop(individual, f)
            except:
                return np.finfo(np.float).max
                
        return np.mean(r_episodes) * -1.0
    
    def _finish_eval(self, individual, X, fitness):
        
        return self._const_opt_eval(individual, f)
    
    def __call__(self, individual):


        self.train_fitness      = partial(self.fitness, action) # ChANGE
        self.X_train            = obs   # CHANGE
        self.y_train            = action    # ChANGE

        '''
            NOTE:
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            optimizer is in const.py as "scipy" : ScipyMinimize
        '''
        
        return self._evaluate_individual(individual) 
    
            
class GPController(gp_base.GPController):
    
    def __init__(self, config_gp_meld, config_task, config_training, name,  action_spec, 
                 algorithm=None, anchor=None, n_episodes=5, env_kwargs=None):
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        self.action_spec                                = action_spec
        
        self._create_model(self, algorithm, anchor)
        
        pset, const_opt, symbolic_actions, action_dim   = self._create_primitive_set(config_task, config_training)                                         
        eval_func                                       = GenericEvaluate(const_opt, name, env_kwargs, symbolic_actions=symbolic_actions, action_dim=action_dim, 
                                                              n_episodes=n_episodes, threshold=threshold) 
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
        #self.symbolic_actions   = [self.tokens_to_DEAP(i, pset) for i in symbolic_actions]
        self.symbolic_actions   = symbolic_actions
            
        return pset, const_opt, symbolic_actions, action_dim

    def _create_toolbox(self, pset, eval_func, max_const=None, constrain_const=False, **kwargs):
                
        toolbox, creator    = self._base_create_toolbox(pset, eval_func, **kwargs) 
        const               = "const" in pset.context
        toolbox             = gp_regression._create_toolbox_const(toolbox, const, max_const)
        
        return toolbox, creator      
        
        