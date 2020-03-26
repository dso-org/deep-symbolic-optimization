"""Factory functions for generating symbolic search tasks."""

from functools import partial

import numpy as np
import gym

from dsr.dataset import Dataset
from dsr.program import Program
from dsr.utils import cached_property


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
        Reward function mapping program.Program object to scalar.

    function_set : list
        List of allowable functions

    n_input_var : int
        Number of input variables
    """

    task_dict = {
        "regression" : make_regression_task,
        # "control" : make_control_task
    }
    
    return task_dict[name](**config_task)


def make_regression_task(metric, metric_params, dataset):
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
    metric = make_regression_metric(metric, y_train, *metric_params)


    def regression_reward(p, test=False, noiseless=False):        

        # Select train or test data, noiseless or not
        X = X_test if test else X_train
        if noiseless:
            y = y_test_noiseless if test else y_train_noiseless
        else:            
            y = y_test if test else y_train

        # Compute estimated values
        y_hat = p.execute(X)

        # Return metric
        r = metric(y, y_hat)
        return r


    ##### Additional regression-specific functions to be used by Programs #####

    @cached_property
    def nmse(p):
        """
        Evaluates and returns the normalized mean squared error of the
        program on the test set (used as final performance metric).
        """
        y_hat = p.execute(X_test)
        return np.mean((y_test - y_hat)**2) / var_y_test


    @cached_property
    def base_r_noiseless(p):
        return regression_reward(p, test=False, noiseless=True)


    @cached_property
    def base_r_test_noiseless(p):
        return regression_reward(p, test=True, noiseless=True)


    @cached_property
    def r_noiseless(p):
        return regression_reward(p, test=False , noiseless=True) - p.complexity


    @cached_property
    def r_test_noiseless(p):
        return regression_reward(p, test=True, noiseless=True) - p.complexity
    

    # Add to Program to be used as cached properties
    Program.nmse = nmse
    Program.base_r_noiseless = base_r_noiseless
    Program.base_r_test_noiseless = base_r_test_noiseless
    Program.r_noiseless = r_noiseless
    Program.r_test_noiseless = r_test_noiseless

    return regression_reward, dataset.function_set, dataset.n_input_var


# Below: Example code for control task. This does not work.
# def make_control_task(self, function_set, env, anchor, action_spec,
#     n_episodes_train=5, n_episodes_test=1000):
#     """
#     Factory function for reinforcement learning environment episodic rewards.
#     This includes closures for an environment and anchor model.

#     Parameters
#     ----------

#     function_set : list
#         List of allowable functions.

#     env : str
#         Gym environment name.

#     anchor : str
#         Path to anchor model.

#     action_spec : dict
#         Dictionary from action dimension to either None, "anchor", or a list of
#         tokens.

#     n_episodes_train : int
#         Number of episodes to run during training.

#     n_episodes_test : int
#         Number of episodes to run during testing.

#     Returns
#     -------

#     See make_task().
#     """

#     # Define closures for environment and anchor model
#     env = gym.make(env)
#     anchor = LOAD_ANCHOR

#     def gym_reward(p, test=False):

#         # Select number of episodes to run
#         n_episodes = n_episodes_test if test else n_episodes_train
        
#         # Run the episodes and return the average episodic reward
#         r_total = 0.0
#         for i in range(n_episodes):
#             obs = self.env.reset()
#             done = False
#             while not done:
#                 action = self.anchor(obs)
#                 action[self.action_dim] = self.p.execute(obs)
#                 obs, r, done, _ = self.env.step(action)
#                 r_total += r
#         return r_total / n_episodes

#     return gym_reward


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

