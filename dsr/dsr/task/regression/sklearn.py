from multiprocessing import Pool
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from dsr.controller import Controller
from dsr.gp import GPController
from dsr.task import set_task
from dsr.train import learn


class DeepSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn interface for deep symbolic regression.
    """

    def __init__(self, config=None):

        config = {} if config is None else config
        config = defaultdict(dict, config)

        self.config_task = config["task"]
        self.config_training = config["training"]
        self.config_controller = config["controller"]
        self.config_gp_meld = config["gp_meld"]

    def fit(self, X, y):

        # Update the Task
        config_task = deepcopy(self.config_task)
        config_task["task_type"] = "regression"
        config_task["dataset"] = (X, y)

        # Instantiate the Controller
        with tf.Session() as sess:
            controller = Controller(sess, **self.config_controller)

        # Instantiate the GPController
        if self.config_gp_meld.get("run_gp_meld"):
            gp_controller = GPController(self.config_gp_meld,
                                         config_task,
                                         self.config_training)

        # Create the pool and set the Task for each worker
        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None and n_cores_batch > 1:
            pool = Pool(n_cores_batch,
                        initializer=set_task,
                        initargs=(config_task,))

        # Set the Task for the parent process
        set_task(config_task)

        # Train the model
        train_result = learn(controller,
                             pool,
                             gp_controller,
                             **self.config_training)
        self.program_ = train_result["program"]

        return self

    def predict(self, X):

        check_is_fitted(self, "program_")

        return self.program_.execute(X)
