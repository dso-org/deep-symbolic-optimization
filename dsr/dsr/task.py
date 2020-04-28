"""Factory functions for generating symbolic search tasks."""

import numpy as np
import gym

from dsr.dataset import Dataset
from dsr.program import Program, from_tokens
from dsr.utils import cached_property
import dsr.utils as U


def make_task(name, **config_task):
    """
    Factory function for reward function that maps a Progarm to a scalar.

    Parameters
    ----------

    name : str
        Name of task:
        "regression" : Regression task.
        "control" : Episodic reinforcement learning task

    config_task : kwargs
        Task-specific arguments. See specifications of task_dict.

    Returns
    -------

    reward_function : function
        Reward function mapping program.Program object to scalar. Includes
        test argument for train vs test evaluation.

    eval_function : function
        Evaluation function mapping program.Program object to a dict of task-
        specific evaluation metrics (primitives). Special optional key "success"
        is used for determining early stopping during training.

    function_set : list
        List of allowable functions (see functions.py for supported functions).

    n_input_var : int
        Number of input variables.
    """

    # Dictionary from task name to task factory function
    task_dict = {
        "regression" : make_regression_task,
        "control" : make_control_task
    }
    
    reward_function, eval_function, function_set, n_input_var = task_dict[name](**config_task)
    return reward_function, eval_function, function_set, n_input_var


def make_regression_task(metric, metric_params, dataset, threshold=1e-12):
    """
    Factory function for regression rewards. This includes closures for a
    dataset and regression metric (e.g. inverse NRMSE). Also sets regression-
    specific metrics to be used by Programs.

    Parameters
    ----------
   
    metric : str
        Name of reward function metric to use.

    metric_params : list
        List of metric-specific parameters.

    dataset : dict
        Dict of dataset.Dataset kwargs.

    Returns
    -------

    See make_task().
    """
    
    # Define closures for dataset and metric
    dataset = Dataset(**dataset)
    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test
    y_train_noiseless = dataset.y_train_noiseless
    y_test_noiseless = dataset.y_test_noiseless
    var_y_test = np.var(dataset.y_test) # Save time by only computing this once
    var_y_test_noiseless = np.var(dataset.y_test_noiseless) # Save time by only computing this once
    metric = make_regression_metric(metric, y_train, *metric_params)


    def reward(p):

        # Compute estimated values
        y_hat = p.execute(X_train)

        # Return metric
        r = metric(y_train, y_hat)
        return r


    def evaluate(p):

        # Compute predictions on test data
        y_hat = p.execute(X_test)

        # NMSE on test data (used to report final error)
        nmse_test = np.mean((y_test - y_hat)**2) / var_y_test

        # NMSE on noiseless test data (used to determine recovery)
        nmse_test_noiseless = np.mean((y_test_noiseless - y_hat)**2) / var_y_test_noiseless

        # Success is defined by NMSE on noiseless test data below a threshold
        success = nmse_test_noiseless < threshold

        info = {
            "nmse_test" : nmse_test,
            "nmse_test_noiseless" : nmse_test_noiseless,
            "success" : success
        }
        return info


    return reward, evaluate, dataset.function_set, dataset.n_input_var


def make_control_task(function_set, env_name, anchor, action_spec,
    n_episodes_train=5, n_episodes_test=1000, success_score=None, dataset=None):
    """
    Factory function for episodic reward function of a reinforcement learning
    environment with continuous actions. This includes closures for the
    environment, an anchor model, and fixed symbolic actions.

    Parameters
    ----------

    function_set : list
        List of allowable functions.

    env_name : str
        Gym environment name.

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

    See make_task().
    """

    # Define closures for environment and anchor model
    env = gym.make(env_name)

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
        U.load_anchor(anchor, env_name)
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


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.
    """

    if "nmse" in name or "nrmse" in name:
        var_y = np.var(y_train)

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                        0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse" :    (lambda y, y_hat : -np.mean((y - y_hat)**2)/var_y,
                        0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse" :   (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)/var_y),
                        0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + var(y)) when y_hat == mean(y)
        "inv_mse" : (lambda y, y_hat : 1/(1 + np.mean((y - y_hat)**2)),
                        0),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 0.5 when y_hat == mean(y)
        "inv_nmse" :    (lambda y, y_hat : 1/(1 + np.mean((y - y_hat)**2)/var_y),
                        0),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 0.5 when y_hat == mean(y)
        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        0),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction" :    (lambda y, y_hat : np.mean(abs(y - y_hat) < args[0]*abs(y) + args[1]),
                        2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson" :     (lambda y, y_hat : scipy.stats.pearsonr(y, y_hat)[0],
                        0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman" :    (lambda y, y_hat : scipy.stats.spearmanr(y, y_hat)[0],
                        0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "Expected {} reward function parameters; received {}.".format(all_metrics[name][1], len(args))
    metric = all_metrics[name][0]
    return metric

