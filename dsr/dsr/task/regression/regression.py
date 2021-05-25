import numpy as np
import pandas as pd

import dsr
from dsr.library import Library
from dsr.functions import create_tokens
from dsr.task.regression.dataset import BenchmarkDataset


def make_regression_task(name, function_set, dataset, metric="inv_nrmse",
    metric_params=(1.0,), extra_metric_test=None, extra_metric_test_params=(),
    reward_noise=0.0, reward_noise_type="r", threshold=1e-12,
    normalize_variance=False, protected=False):
    """
    Factory function for regression rewards. This includes closures for a
    dataset and regression metric (e.g. inverse NRMSE). Also sets regression-
    specific metrics to be used by Programs.

    Parameters
    ----------
    name : str or None
        Name of regression benchmark, if using benchmark dataset.

    function_set : list or None
        List of allowable functions. If None, uses function_set according to
        benchmark dataset.

    dataset : dict, str, or tuple
        If dict: .dataset.BenchmarkDataset kwargs.
        If str: filename of dataset.
        If tuple: (X, y) data

    metric : str
        Name of reward function metric to use.

    metric_params : list
        List of metric-specific parameters.

    extra_metric_test : str
        Name of extra function metric to use for testing.

    extra_metric_test_params : list
        List of metric-specific parameters for extra test metric.

    reward_noise : float
        Noise level to use when computing reward.

    reward_noise_type : "y_hat" or "r"
        "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
        "r" : N(0, reward_noise) is added to r.

    normalize_variance : bool
        If True and reward_noise_type=="r", reward is multiplied by
        1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

    protected : bool
        Whether to use protected functions.

    threshold : float
        Threshold of NMSE on noiseless data used to determine success.

    Returns
    -------

    task : Task
        Dynamically created Task object whose methods contains closures.
    """

    X_test = y_test = y_test_noiseless = None

    # Benchmark dataset config
    if isinstance(dataset, dict):
        dataset["name"] = name
        benchmark = BenchmarkDataset(**dataset)
        X_train = benchmark.X_train
        y_train = benchmark.y_train
        X_test = benchmark.X_test
        y_test = benchmark.y_test
        y_test_noiseless = benchmark.y_test_noiseless

        # Unless specified, use the benchmark's default function_set
        if function_set is None:
            function_set = benchmark.function_set

    # Dataset filename
    elif isinstance(dataset, str):
        df = pd.read_csv(dataset, header=None) # Assuming data file does not have header rows
        X_train = df.values[:, :-1]
        y_train = df.values[:, -1]

    # sklearn-like (X, y) data
    elif isinstance(dataset, tuple):
        X_train = dataset[0]
        y_train = dataset[1]

    if X_test is None:
        X_test = X_train
        y_test = y_train
        y_test_noiseless = y_test

    if function_set is None:
        print("WARNING: Function set not provided. Using default set.")
        function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]

    # Save time by only computing these once
    var_y_test = np.var(y_test)
    var_y_test_noiseless = np.var(y_test_noiseless)

    # Define closures for metric
    metric, invalid_reward, max_reward = make_regression_metric(metric, y_train, *metric_params)
    if extra_metric_test is not None:
        print("Setting extra test metric to {}.".format(extra_metric_test))
        metric_test, _, _ = make_regression_metric(extra_metric_test, y_test, *extra_metric_test_params) 
    assert reward_noise >= 0.0, "Reward noise must be non-negative."
    if reward_noise:
        assert reward_noise_type in ["y_hat", "r"], "Reward noise type not recognized."
        rng = np.random.RandomState(0)
        y_rms_train = np.sqrt(np.mean(y_train ** 2))
        if reward_noise_type == "y_hat":
            scale = reward_noise * y_rms_train
        elif reward_noise_type == "r":
            scale = reward_noise

    def reward(p):

        # Compute estimated values
        y_hat = p.execute(X_train)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            return invalid_reward

        ### Observation noise
        # For reward_noise_type == "y_hat", success must always be checked to 
        # ensure success cases aren't overlooked due to noise. If successful,
        # return max_reward.
        if reward_noise and reward_noise_type == "y_hat":
            if p.evaluate.get("success"):
                return max_reward
            y_hat += rng.normal(loc=0, scale=scale, size=y_hat.shape)

        # Compute metric
        r = metric(y_train, y_hat)

        ### Direct reward noise
        # For reward_noise_type == "r", success can for ~max_reward metrics be
        # confirmed before adding noise. If successful, must return np.inf to
        # avoid overlooking success cases.
        if reward_noise and reward_noise_type == "r":
            if r >= max_reward - 1e-5 and p.evaluate.get("success"):
                return np.inf
            r += rng.normal(loc=0, scale=scale)
            if normalize_variance:
                r /= np.sqrt(1 + 12*scale**2)

        return r


    def evaluate(p):

        # Compute predictions on test data
        y_hat = p.execute(X_test)
        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
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

        if extra_metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = metric_test(y_test, y_hat)
                m_test_noiseless = metric_test(y_test_noiseless, y_hat)     

            info.update(
                {
                extra_metric_test : m_test,
                extra_metric_test + '_noiseless' : m_test_noiseless
                }
            )

        return info

    tokens = create_tokens(n_input_var=X_train.shape[1],
                           function_set=function_set,
                           protected=protected)
    library = Library(tokens)

    stochastic = reward_noise > 0.0

    extra_info = {}

    task = dsr.task.Task(reward_function=reward,
                evaluate=evaluate,
                library=library,
                stochastic=stochastic,
                extra_info=extra_info)

    return task


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

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    var_y = np.var(y_train)

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                        0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse" :     (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)),
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

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse" : (lambda y, y_hat : -np.log(1 + np.mean((y - y_hat)**2)),
                        0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse" : (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)),
                        1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)/var_y),
                        1),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        1),

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
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse" : -var_y,
        "neg_rmse" : -np.sqrt(var_y),
        "neg_nmse" : -1.0,
        "neg_nrmse" : -1.0,
        "neglog_mse" : -np.log(1 + var_y),
        "inv_mse" : 0.0, #1/(1 + args[0]*var_y),
        "inv_nmse" : 0.0, #1/(1 + args[0]),
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        "fraction" : 0.0,
        "pearson" : 0.0,
        "spearman" : 0.0
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "neg_mse" : 0.0,
        "neg_rmse" : 0.0,
        "neg_nmse" : 0.0,
        "neg_nrmse" : 0.0,
        "neglog_mse" : 0.0,
        "inv_mse" : 1.0,
        "inv_nmse" : 1.0,
        "inv_nrmse" : 1.0,
        "fraction" : 1.0,
        "pearson" : 1.0,
        "spearman" : 1.0
    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward
