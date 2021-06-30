"""Core deep symbolic optimizer construct."""

import json
from collections import defaultdict
from multiprocessing import Pool
import random
from time import time

import numpy as np
import tensorflow as tf

from dsr.task import set_task
from dsr.controller import Controller
from dsr.train import learn
from dsr.prior import make_prior
from dsr.program import Program


class DeepSymbolicOptimizer():
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON.

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
        self.sess = None

    def setup(self):

        # Clear the cache, reset the compute graph, and set seeds
        Program.clear_cache()
        tf.reset_default_graph()
        self.set_seeds() # Must be called _after_ resetting graph

        self.pool = self.make_pool()
        self.sess = tf.Session()
        self.prior = self.make_prior()
        self.controller = self.make_controller()
        self.gp_controller = self.make_gp_controller()

    def train(self):

        # Setup the model
        self.setup()

        # Train the model
        result = learn(self.sess,
                       self.controller,
                       self.pool,
                       self.gp_controller,
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
        self.config_prior = self.config["prior"]
        self.config_training = self.config["training"]
        self.config_controller = self.config["controller"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]

    def set_seeds(self):
        """
        Set the tensorflow, numpy, and random module seeds based on the seed
        specified in config. If there is no seed or it is None, a time-based
        seed is used instead and is written to config.
        """

        seed = self.config_experiment.get("seed")
        if seed is None:
            # Default uses current time in milliseconds, modulo 1e9
            seed = round(time() * 1000) % int(1e9)
            self.config_experiment["seed"] = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_controller(self):
        controller = Controller(self.sess,
                                self.prior,
                                **self.config_controller)
        return controller

    def make_gp_controller(self):
        if self.config_gp_meld.pop("run_gp_meld", False):
            from dsr.gp.gp_controller import GPController
            gp_controller = GPController(self.prior,
                                         **self.config_gp_meld)
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

    def save(self, save_path):

        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):

        if self.sess is None:
            self.setup()
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
