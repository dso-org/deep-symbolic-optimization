import gym
import numpy as np

from dsr.program import from_tokens
from . import utils as U


def make_control_task(function_set, name, action_spec, algorithm=None,
    anchor=None, n_episodes_train=5, n_episodes_test=1000, success_score=None):
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
        anchor for given environment.

    anchor : str or None
        Path to anchor model, or None to use default anchor for given
        environment.

    n_episodes_train : int
        Number of episodes to run during training.

    n_episodes_test : int
        Number of episodes to run during testing.

    Returns
    -------

    See dsr.task.task.make_task().
    """

    # Define closures for environment and anchor model
    env = gym.make(name)

    # Configuration assertions
    assert len(env.observation_space.shape) == 1, "Only support vector observation spaces."
    assert isinstance(env.action_space, gym.spaces.Box), "Only supports continuous action spaces."
    n_actions = env.action_space.shape[0]
    assert n_actions == len(action_spec), "Received specifications for {} action dimensions; expected {}.".format(len(action_spec), n_actions)
    assert len([v for v in action_spec if v is None]) == 1, "Exactly 1 action_spec element must be None."
    assert int(algorithm is None) + int(anchor is None) in [0, 2], "Either none or both of (algorithm, anchor) must be None."

    # Load the anchor model (if applicable)
    if "anchor" in action_spec:
        # Load custom anchor, if provided, otherwise load default
        if algorithm is not None and anchor is not None:
            U.load_model(algorithm, anchor_path)
        else:
            U.load_default_model(name)

    # Generate symbolic policies and determine action dimension
    symbolic_actions = {}
    for i, spec in enumerate(action_spec):

        # Action dimnension being learned
        if spec is None:
            action_dim = i

        # Pre-specified symbolic policy
        elif isinstance(spec, list):
            tokens = None # TBD: Convert str to ints
            p = from_tokens(tokens, optimize=False)
            symbolic_actions[i] = p

        else:
            assert spec == "anchor", "Action specifications must be None, a list of tokens, or 'anchor'."


    def get_action(p, obs):
        """Helper function to get an action from Program p according to obs,
        since Program.execute() requires 2D arrays but we only want 1D."""
        
        return p.execute(np.array([obs]))[0]


    def run_episodes(p, n_episodes):
        """Runs n_episodes episodes and returns each episodic reward."""

        # Run the episodes and return the average episodic reward
        r_episodes = np.zeros(n_episodes, dtype=np.float32) # Episodic rewards for each episode
        for i in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:

                # Compute anchor actions
                if U.model is not None:
                    action, _ = U.model.predict(obs)
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)

                # Replace fixed symbolic actions
                for i, fixed_p in symbolic_actions.items():
                    action[i] = get_action(fixed_p, obs)

                # Replace symbolic action with current program
                action[action_dim] = get_action(p, obs)
                
                obs, r, done, _ = env.step(action)
                r_episodes[i] += r

        return r_episodes


    def reward(p):

        # Run the episodes
        r_episodes = run_episodes(p, n_episodes_train)

        # Return the mean
        r_avg = np.mean(r_episodes)
        return r_avg


    def evaluate(p):

        # Run the episodes
        r_episodes = run_episodes(p, n_episodes_test)

        # Compute eval statistics
        r_avg_test = np.mean(r_episodes)
        success_rate = np.mean(r_episodes > success_score)
        success = success_rate == 1.0

        info = {
            "r_avg_test" : r_avg_test,
            "success_rate" : success_rate,
            "success" : success
        }
        return info


    n_input_var = env.observation_space.shape[0]

    return reward, evaluate, function_set, n_input_var
