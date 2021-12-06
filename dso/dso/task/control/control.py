import gym
import numpy as np

import dso.task.control # Registers custom and third-party environments
from dso.program import Program, from_str_tokens
from dso.library import Library
from dso.functions import create_tokens
import dso.task.control.utils as U
from dso.task import HierarchicalTask


REWARD_SEED_SHIFT = int(1e6) # Reserve the first million seeds for evaluation

# Pre-computed values for reward scale
REWARD_SCALE = {
    "CustomCartPoleContinuous-v0" : [0.0, 1000.0],
    "MountainCarContinuous-v0" : [0.0, 93.95],
    "Pendulum-v0" : [-1300.0, -147.56],
    "InvertedDoublePendulumBulletEnv-v0" : [0.0, 9357.77],
    "InvertedPendulumSwingupBulletEnv-v0" : [0.0, 891.34],
    "LunarLanderContinuous-v2" : [0.0, 272.65],
    "HopperBulletEnv-v0" : [0.0, 2741.86],
    "ReacherBulletEnv-v0" : [-5.0, 19.05],
    "BipedalWalker-v2" : [-60.0, 312.0]
}


class ControlTask(HierarchicalTask):
    """
    Class for the control task. Discrete objects are expressions, which are
    evaluated by directly using them as control policies in a reinforcement
    learning environment.
    """

    def __init__(self, function_set, env, action_spec, algorithm=None,
                 anchor=None, n_episodes_train=5, n_episodes_test=1000,
                 success_score=None, protected=False, env_kwargs=None,
                 fix_seeds=False, episode_seed_shift=0, reward_scale=True,
                 multiobject=False, decision_tree_threshold_set=None):
        """
        Parameters
        ----------

        function_set : list
            List of allowable functions.

        env : str
            Name of Gym environment, e.g. "Pendulum-v0" or "my_module:MyEnv-v0".

        action_spec : list
            List of action specifications: None, "anchor", or a list of tokens.

        algorithm : str or None
            Name of algorithm corresponding to anchor path, or None to use
            default anchor for given environment.

        anchor : str or None
            Path to anchor model, or None to use default anchor for given
            environment.

        n_episodes_train : int
            Number of episodes to run during training.

        n_episodes_test : int
            Number of episodes to run during testing.

        success_score : float
            Episodic reward considered to be "successful." A Program will have
            success=True if all n_episodes_test episodes achieve this score.

        protected : bool
            Whether or not to use protected operators.

        env_kwargs : dict
            Dictionary of environment kwargs passed to gym.make().

        fix_seeds : bool
            If True, environment uses the first n_episodes_train seeds for
            reward and the next n_episodes_test seeds for evaluation. This makes
            the task deterministic.

        episode_seed_shift : int
            Training episode seeds start at episode_seed_shift * 100 +
            REWARD_SEED_SHIFT. This has no effect if fix_seeds == False.

        multiobject : bool
            Whether learning a multi-object Program. If True, ignores action
            spec and generates expressions for all action dimensions.

        reward_scale : list or bool
            If list: list of [r_min, r_max] used to scale rewards. If True, use
            default values in REWARD_SCALE. If False, don't scale rewards.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision
            trees.
        """

        super(HierarchicalTask).__init__()

        # Set member variables used by member functions
        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.success_score = success_score
        self.fix_seeds = fix_seeds
        self.episode_seed_shift = episode_seed_shift
        self.multiobject = multiobject
        self.stochastic = not fix_seeds

        # Create the environment
        env_name = env
        if env_kwargs is None:
            env_kwargs = {}
        self.env = gym.make(env_name, **env_kwargs)

        # Note Zoo is not implemented as a package, which might make this tedious
        if "Bullet" in env_name:
            self.env = U.TimeFeatureWrapper(self.env)

        # Determine reward scaling
        if isinstance(reward_scale, list):
            assert len(reward_scale) == 2, "Reward scale should be length 2: \
                                            min, max."
            self.r_min, self.r_max = reward_scale
        elif reward_scale:
            if env_name in REWARD_SCALE:
                self.r_min, self.r_max = REWARD_SCALE[env_name]
            else:
                raise RuntimeError("{} has no default values for reward_scale. \
                                   Use reward_scale=False or specify \
                                   reward_scale=[r_min, r_max]."
                                   .format(env_name))
        else:
            self.r_min = self.r_max = None

        # Set the library (do this now in case there are symbolic actions)
        n_input_var = self.env.observation_space.shape[0]
        tokens = create_tokens(n_input_var, function_set, protected,
                               decision_tree_threshold_set)
        self.library = Library(tokens)
        Program.library = self.library

        # Configuration assertions
        assert len(self.env.observation_space.shape) == 1, \
               "Only support vector observation spaces."
        assert isinstance(self.env.action_space, gym.spaces.Box), \
               "Only supports continuous action spaces."
        n_actions = self.env.action_space.shape[0]
        if multiobject and n_actions == 1:
            print("WARNING: Setting multiobject=True has no effect for \
                  environment with 1 action dimension.")
        if multiobject:
            Program.set_n_objects(n_actions)
        assert n_actions == len(action_spec), "Received spec for {} action \
               dimensions; expected {}.".format(len(action_spec), n_actions)
        if not multiobject:
            assert (len([v for v in action_spec if v is None]) <= 1), \
                   "No more than 1 action_spec element can be None."
        else:
            print("WARNING: multiobject=True; action_spec will be ignored.")
        assert int(algorithm is None) + int(anchor is None) in [0, 2], \
               "Either none or both of (algorithm, anchor) must be None."

        # Load the anchor model (if applicable)
        if "anchor" in action_spec:
            # Load custom anchor, if provided, otherwise load default
            if algorithm is not None and anchor is not None:
                U.load_model(algorithm, anchor)
            else:
                U.load_default_model(env_name)
            self.model = U.model
        else:
            self.model = None

        # Generate symbolic policies and determine action dimension
        self.symbolic_actions = {}
        self.action_dim = None
        if not multiobject:
            for i, spec in enumerate(action_spec):

                # Action taken from anchor policy
                if spec == "anchor":
                    continue

                # Action dimnension being learned
                if spec is None:
                    self.action_dim = i

                # Pre-specified symbolic policy
                elif isinstance(spec, list) or isinstance(spec, str):
                    str_tokens = spec
                    p = from_str_tokens(str_tokens, optimize=False,
                                        skip_cache=True)
                    self.symbolic_actions[i] = p
                else:
                    assert False, "Action specifications must be None, a \
                                  str/list of tokens, or 'anchor'."
        else:
            self.action_dim = list(range(n_actions))

        # Define name based on environment and learned action dimension
        self.name = env_name
        if self.action_dim is not None:
            self.name += "_a{}".format(self.action_dim)

    def get_action(self, p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""

        action = p.execute(np.array([obs]))[0]
        return np.asarray(action)

    def run_episodes(self, p, n_episodes, evaluate):
        """Runs n_episodes episodes and returns each episodic reward."""

        # Run the episodes and return the average episodic reward
        r_episodes = np.zeros(n_episodes, dtype=np.float64) # Episodic rewards for each episode
        for i in range(n_episodes):

            # During evaluation, always use the same seeds
            if evaluate:
                self.env.seed(i)
            elif self.fix_seeds:
                seed = i + (self.episode_seed_shift * 100) + REWARD_SEED_SHIFT
                self.env.seed(seed)
            obs = self.env.reset()
            done = False
            while not done:

                # Compute anchor actions
                if self.model is not None:
                    action, _ = self.model.predict(obs)
                else:
                    action = np.zeros(self.env.action_space.shape,
                                      dtype=np.float32)

                # Replace fixed symbolic actions
                for j, fixed_p in self.symbolic_actions.items():
                    action[j] = self.get_action(fixed_p, obs)

                # Replace symbolic action with current program
                if self.action_dim is not None:
                    if self.multiobject:
                        action = self.get_action(p, obs)
                    else:
                        action[self.action_dim] = self.get_action(p, obs)

                # Replace NaNs and clip infinites
                action[np.isnan(action)] = 0.0 # Replace NaNs with zero
                action = np.clip(action,
                                 self.env.action_space.low,
                                 self.env.action_space.high)

                obs, r, done, _ = self.env.step(action)
                r_episodes[i] += r

        return r_episodes

    def reward_function(self, p):

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_train, evaluate=False)

        # Return the mean
        r_avg = np.mean(r_episodes)

        # Scale rewards to [0, 1]
        if self.r_min is not None:
            r_avg = (r_avg - self.r_min) / (self.r_max - self.r_min)

        return r_avg

    def evaluate(self, p):

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_test, evaluate=True)

        # Compute eval statistics
        r_avg_test = np.mean(r_episodes)
        success_rate = np.mean(r_episodes >= self.success_score)
        success = success_rate == 1.0

        info = {
            "r_avg_test" : r_avg_test,
            "success_rate" : success_rate,
            "success" : success
        }
        return info
