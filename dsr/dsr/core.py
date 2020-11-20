"""Core deep symbolic optimizer construct."""

import json
import zlib
from collections import defaultdict
from multiprocessing import Pool

import tensorflow as tf

from dsr.task import set_task
from dsr.controller import Controller
from dsr.gp import GPController
from dsr.train import learn
from dsr.program import Program


class DeepSymbolicOptimizer():
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON. See dsr/dsr/config.json for template.

    Attributes
    ----------
    config : dict
        Configuration parameters for training.

    Methods
    -------
    train
        Builds and trains the model according to config.
    """

    def __init__(self, config=None):
        self.update_config(config)

    def train(self, seed=0):

        # Clear the cache, reset the compute graph, and set the seed
        Program.clear_cache()
        tf.reset_default_graph()
        self.seed(seed) # Must be called _after_ resetting graph

        pool = self.make_pool()
        sess = tf.Session()
        controller = self.make_controller(sess)
        gp_controller = self.make_gp_controller()

        # Train the model
        result = learn(sess,
                       controller,
                       pool,
                       gp_controller,
                       **self.config_training)
        return result

    def update_config(self, config):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'rb') as f:
                config = json.load(f)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_training = self.config["training"]
        self.config_controller = self.config["controller"]
        self.config_gp_meld = self.config["gp_meld"]

    def seed(self, seed_=0):
        """Set the tensorflow seed, which will be offset by a checksum on the
        task name to ensure seeds differ across different tasks."""

        if "name" in self.config_task:
            task_name = self.config_task["name"]
        else:
            task_name = ""
        seed_ += zlib.adler32(task_name.encode("utf-8"))
        tf.set_random_seed(seed_)

        return seed_

    def make_controller(self, sess):
        controller = Controller(sess, **self.config_controller)
        return controller

    def make_gp_controller(self):
        if self.config_gp_meld.get("run_gp_meld"):
            gp_controller = GPController(self.config_gp_meld,
                                         self.config_task,
                                         self.config_training)
        else:
            gp_controller = None
        return gp_controller

    def make_pool(self):
        # Create the pool and set the Task for each worker
        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None and n_cores_batch > 1:
            pool = Pool(n_cores_batch,
                        initializer=set_task,
                        initargs=(self.config_task,))

        # Set the Task for the parent process
        set_task(self.config_task)

        return pool
