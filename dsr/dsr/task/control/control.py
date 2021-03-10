import gym
import multiprocessing
from multiprocessing import Pool
import random

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

import numpy as np

import dsr
from dsr.program import Program, from_str_tokens
from dsr.library import Library
from dsr.functions import create_tokens
from . import utils as U


REWARD_SEED_SHIFT   = int(1e6) # Reserve the first million seeds for evaluation


def _get_action(p, obs):
    """Helper function to get an action from Program p according to obs,
    since Program.execute() requires 2D arrays but we only want 1D."""

    action = p.execute(np.array([obs]))[0]

    return action


def episode(p_input, action_dim, evaluate, fix_seeds, model, episode_seed_shift,  symbolic_actions, env, seed,
            reward_seed_shift=REWARD_SEED_SHIFT, get_action=_get_action, get_fixed_action=_get_action):

    r_episode   = 0.0

    assert callable(get_action)
    assert callable(get_fixed_action)
    assert seed is not None
    
    # When we use the parallel version, we have to copy the program over the hard way because 
    # Python has trouble pickling an existing program. This gets thrown away at the end of the 
    # run here, so we don't effect the actually orignal program.
    if action_dim is not None:
        if isinstance(p_input,dsr.program.Program):
            p = p_input
        elif callable(p_input):
            p = p_input
        else:
            p = Program(p_input, False)

    # During evaluation, always use the same seeds
    if evaluate:
        env.seed(seed)
    elif fix_seeds:
        env.seed(seed + (episode_seed_shift * 100) + reward_seed_shift)
        
    obs = env.reset()
    done = False
    while not done:

        # Compute anchor actions
        if model is not None:
            action, _ = model.predict(obs)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Replace fixed symbolic actions
        for j, fixed_p in symbolic_actions.items():
            action[j] = get_fixed_action(fixed_p, obs)

        # Replace symbolic action with current program
        if action_dim is not None:
            action[action_dim] = get_action(p, obs)
        
        # Replace NaNs and clip infinites
        action[np.isnan(action)] = 0.0 # Replace NaNs with zero
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, r, done, _ = env.step(action)
        r_episode += r
    
    return r_episode


def create_model(action_spec, algorithm, anchor, anchor_path=None):
    
    # Load the anchor model (if applicable)
    if "anchor" in action_spec:
        # Load custom anchor, if provided, otherwise load default
        if algorithm is not None and anchor is not None:
            U.load_model(algorithm, anchor_path)
        else:
            U.load_default_model(name)
        model = U.model
    else:
        model = None

    return model


def create_symbolic_actions(action_spec):
    
    symbolic_actions = {}
    action_dim = None
    for i, spec in enumerate(action_spec):

        # Action taken from anchor policy
        if spec == "anchor":
            continue

        # Action dimnension being learned
        if spec is None:
            action_dim = i

        # Pre-specified symbolic policy
        elif isinstance(spec, list) or isinstance(spec, str):
            str_tokens = spec
            symbolic_actions[i] = from_str_tokens(str_tokens, optimize=False, skip_cache=True)
        else:
            assert False, "Action specifications must be None, a str/list of tokens, or 'anchor'."

    return symbolic_actions, action_dim


def make_env(name, env_kwargs):
    
    assert "Bullet" not in name or pybullet_envs is not None, "Must install pybullet_envs."
    if env_kwargs is None:
        env_kwargs = {}

    # Define closures for environment and anchor model
    env = gym.make(name, **env_kwargs)

    # HACK: Wrap pybullet envs in TimeFeatureWrapper
    # TBD: Load the Zoo hyperparameters, including wrapper features, not just the model.
    # Note Zoo is not implemented as a package, which might make this tedious
    if "Bullet" in name:
        env = U.TimeFeatureWrapper(env)
        
    return env, env_kwargs

'''
    We need a wrapper since if the pool is inside the control task itself, Python gets upset when 
    we use n_cores_batch since it sees a pool of processes within a pool. This tucks it aside. 
    so we can parallelize gym runs for long validation and evaluation, but use batch pooling
    for shorter tests.
'''
class pool_wrapper:

    def __init__(self):
        self.pool = None
        
    def __call__(self):
        if self.pool is None:
            self.pool = Pool(processes = multiprocessing.cpu_count())
            
        return self.pool


def make_control_task(function_set, name, action_spec, algorithm=None,
    anchor=None, n_episodes_slice=None, n_episodes_train=5, n_episodes_validate=100, n_episodes_long_validate=1000, n_episodes_test=1000, 
    success_score=None, stochastic=True, protected=False, env_kwargs=None, fix_seeds=False,
    episode_seed_shift=0, do_validate=False, do_long_validate=False, long_validation_finalists=3, slice_optimize_stat="min"):
    """
    Factory function for episodic reward function of a reinforcement learning
    environment with continuous actions. This includes closures for the
    environment, an anchor model, and fixed symbolic actions.

    Parameters
    ----------

    function_set : list
        List of allowable functions.

    name : str
        Name of Gym environment.

    action_spec : list
        List of action specifications: None, "anchor", or a list of tokens.

    algorithm : str or None
        Name of algorithm corresponding to anchor path, or None to use default
        anchor for given environment. For example see "stable-baselines".

    anchor : str or None
        Path to anchor model, or None to use default anchor for given
        environment.

    n_episodes_train : int
        Number of episodes to run during training.

    n_episodes_test : int
        Number of episodes to run during testing.

    stochastic : bool
        If True, Programs will not be cached, and thus identical traversals will
        be evaluated as unique objects. The hall of fame will be based on the
        average reward seen for each unique traversal.

    protected : bool
        Whether or not to use protected operators.

    env_kwargs : dict
        Dictionary of environment kwargs passed to gym.make().

    fix_seeds : bool
        If True, environment uses the first n_episodes_train seeds for reward
        and the next n_episodes_test seeds for evaluation. This makes the task
        deterministic.

    episode_seed_shift : int
        Training episode seeds start at episode_seed_shift * 100 +
        REWARD_SEED_SHIFT. This has no effect if fix_seeds == False.

    Returns
    -------

    See dsr.task.task.make_task().
    """

    env, env_kwargs = make_env(name, env_kwargs)

    # Set the library (need to do this now in case there are symbolic actions)
    if fix_seeds and stochastic:
        print("WARNING: fix_seeds=True renders task deterministic. Overriding to stochastic=False.")
        stochastic = False
    n_input_var = env.observation_space.shape[0]
    tokens = create_tokens(n_input_var, function_set, protected)
    library = Library(tokens)
    Program.library = library

    # Configuration assertions
    assert len(env.observation_space.shape) == 1, "Only support vector observation spaces."
    assert isinstance(env.action_space, gym.spaces.Box), "Only supports continuous action spaces."
    n_actions = env.action_space.shape[0]
    assert n_actions == len(action_spec), "Received specifications for {} action dimensions; expected {}.".format(len(action_spec), n_actions)
    assert len([v for v in action_spec if v is None]) <= 1, "No more than 1 action_spec element can be None."
    assert int(algorithm is None) + int(anchor is None) in [0, 2], "Either none or both of (algorithm, anchor) must be None."

    model                   = create_model(action_spec, algorithm, anchor, anchor_path=None)
    gym_pool                = pool_wrapper()
    #val_extra_seed_shift    = 0

    # Generate symbolic policies and determine action dimension
    symbolic_actions, action_dim    = create_symbolic_actions(action_spec)

    def get_action(p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""

        action = _get_action(p, obs)

        return action

    def run_episodes(p, n_episodes, evaluate, extra_seed_shift=0):
        """Runs n_episodes episodes and returns each episodic reward."""

        # Run the episodes and return the average episodic reward
        r_episodes = np.zeros(n_episodes, dtype=np.float64) # Episodic rewards for each episode
        
        for i in range(n_episodes):
            r_episodes[i] = episode(p, action_dim, evaluate, fix_seeds, model, episode_seed_shift+extra_seed_shift, symbolic_actions, env, seed=i)

        return r_episodes

    def par_run_episodes(p, n_episodes, evaluate, extra_seed_shift=0):
        """Runs n_episodes episodes and returns each episodic reward.
        
        Parallel version for when we have just a few individuals, but lots of test oer. Thus we don't have batch or task paralellization. 
        """       
        r_episodes          = np.zeros(n_episodes, dtype=np.float64) # Episodic rewards for each episode
        
        seed_shift          = episode_seed_shift+extra_seed_shift
        par_fix_seeds       = True      # We alllllways need to seed here otherwise each process runs just like the other. 
        
        multiple_results    = [gym_pool().apply_async(episode, (p.tokens, action_dim, evaluate, par_fix_seeds, model, seed_shift, 
                                                               symbolic_actions, env, seed)) for seed in range(n_episodes)]
                
        vals                = [res.get() for res in multiple_results]
        for i,v in enumerate(vals):
            r_episodes[i]   = v  # Ordering probably does not matter here. 
            
        return r_episodes
        
    def reward(p):

        # Run the episodes
        r_episodes = run_episodes(p, n_episodes_train, evaluate=False)

        # Return the mean
        r_avg = np.mean(r_episodes)
        
        return r_avg

    def validate(p):

        # Use new seeds never in reward
        # Add in a random number so we are not ALWAYS using the same seed numbers. Only evaluate should do this. 
        extra_seed_shift    = n_episodes_train + random.randint(0,1e9)

        # Run the episodes
        r_episodes          = par_run_episodes(p, n_episodes_validate, evaluate=False, extra_seed_shift=extra_seed_shift)

        # Compute val statistics
        v_r_avg = np.mean(r_episodes)

        return v_r_avg
    
    def long_validate(p):

        # Use new seeds never in reward or short validate
        # Add in a random number so we are not ALWAYS using the same seed numbers. Only evaluate should do this. 
        extra_seed_shift    = n_episodes_train + n_episodes_validate + int(1e9) + random.randint(0,1e9)

        # Run the episodes        
        r_episodes          = par_run_episodes(p, n_episodes_long_validate, evaluate=False, extra_seed_shift=extra_seed_shift)
        
        # Compute val statistics
        lv_r_avg = np.mean(r_episodes)

        return lv_r_avg

    def evaluate(p):

        # Run the episodes
        r_episodes = par_run_episodes(p, n_episodes_test, evaluate=True)

        # Compute eval statistics
        r_avg_test = np.mean(r_episodes)
        success_rate = np.mean(r_episodes >= success_score)
        success = success_rate == 1.0
        
        info = {
            "r_avg_test" : r_avg_test,
            "success_rate" : success_rate,
            "success" : success,
            "test_val" : r_avg_test
        }
        return info

    extra_info = {
        "symbolic_actions"          : symbolic_actions,
        "do_validate"               : do_validate,
        "do_long_validate"          : do_long_validate,
        "long_validation_finalists" : long_validation_finalists
    }
    

    task = dsr.task.Task(reward_function=reward,
                         validate_function=validate,
                         long_validate_function=long_validate,
                         evaluate=evaluate,
                         library=library,
                         stochastic=stochastic,
                         extra_info=extra_info)

    return task
