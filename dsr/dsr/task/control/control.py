import gym
import numpy as np

from dsr.program import Program, from_tokens
from . import utils as U


def make_control_task(function_set, name, anchor, action_spec,
    n_episodes_train=5, n_episodes_test=1000, success_score=None, dataset=None):
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

    anchor : str or None
        Path to anchor model, or None if not using an anchor.

    action_spec : dict
        Dictionary from action dimension to either None, "anchor", or a list of
        tokens.

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
    assert len([v for v in action_spec.values() if v is None]) == 1, "Exactly 1 action_spec value must be None."
    int_keys = [int(k.split('_')[-1]) for k in action_spec.keys()]
    assert set(int_keys) == set(range(n_actions)), "Expected keys ending with 0, 1, ..., n_actions."

    # Replace action_spec with ordered list
    for k in list(action_spec.keys()):
        int_key = int(k.split('_')[-1])
        action_spec[int_key] = action_spec.pop(k)
    action_spec = [action_spec[i] for i in range(n_actions)] 

    # Load the anchor model (if applicable)
    if "anchor" in action_spec:
        assert anchor is not None, "At least one action uses anchor, but anchor model not specified."
        U.load_anchor(anchor, name)
        anchor = U.model
    else:
        anchor = None

    # Generate symbolic policies and determine action dimension
    symbolic_actions = {}
    for i, spec in enumerate(action_spec):

        # Action dimnension being learned
        if spec is None:
            action_dim = i

        # Pre-specified symbolic policy
        elif isinstance(spec, list):
            tokens = None # Convert str to ints
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
                if anchor is not None:
                    action, _ = anchor.predict(obs)
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