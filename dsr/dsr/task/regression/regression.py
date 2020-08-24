import numpy as np

from .dataset import Dataset


def make_regression_task(name, metric, metric_params, dataset, threshold=1e-12):
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
        Dict of .dataset.Dataset kwargs.

    Returns
    -------

    See dsr.task.task.make_task().
    """
    
    # Define closures for dataset and metric
    dataset["name"] = name # TBD: Refactor to not have two instances of "name"
    dataset = Dataset(**dataset)
    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test
    y_train_noiseless = dataset.y_train_noiseless
    y_test_noiseless = dataset.y_test_noiseless
    var_y_test = np.var(dataset.y_test) # Save time by only computing this once
    var_y_test_noiseless = np.var(dataset.y_test_noiseless) # Save time by only computing this once
    metric, invalid_reward = make_regression_metric(metric, y_train, *metric_params)


    def reward(p):

        # Compute estimated values
        y_hat = p.execute(X_train)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            r = invalid_reward            

        # Otherwise, return metric
        else:
            r = metric(y_train, y_hat)

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
        return info

    stochastic = False # Regression rewards are deterministic


    return reward, evaluate, dataset.function_set, dataset.n_input_var, stochastic


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
    """

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
    assert len(args) == all_metrics[name][1], "Expected {} reward function parameters; received {}.".format(all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, inavlid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse" : -var_y,
        "neg_nmse" : -1.0,
        "neg_nrmse" : -1.0,
        "inv_mse" : 0.0, #1/(1 + args[0]*var_y),
        "inv_nmse" : 0.0, #1/(1 + args[0]),
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        "fraction" : 0.0,
        "pearson" : 0.0,
        "spearman" : 0.0
    }
    invalid_reward = all_invalid_rewards[name]

    return metric, invalid_reward
