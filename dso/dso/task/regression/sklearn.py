from copy import deepcopy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from dso import DeepSymbolicOptimizer


class DeepSymbolicRegressor(DeepSymbolicOptimizer,
                            BaseEstimator, RegressorMixin):
    """
    Sklearn interface for deep symbolic regression.
    """

    def __init__(self, config=None):
        if config is None:
            config = {
                "task" : {"task_type" : "regression"}
            }
        DeepSymbolicOptimizer.__init__(self, config)

    def fit(self, X, y):

        # Update the Task
        config = deepcopy(self.config)
        config["task"]["dataset"] = (X, y)

        # Turn off file saving
        config["experiment"]["logdir"] = None

        self.set_config(config)

        train_result = self.train()
        self.program_ = train_result["program"]

        return self

    def predict(self, X):

        check_is_fitted(self, "program_")

        return self.program_.execute(X)
