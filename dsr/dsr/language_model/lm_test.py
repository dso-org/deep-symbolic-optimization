"""
language model comm test

"""

import os
import sys
import json
import multiprocessing
from itertools import compress
from datetime import datetime
from textwrap import indent

import tensorflow as tf
import pandas as pd
import numpy as np

from dsr.controller import Controller
from dsr.program import Program, from_tokens
from dsr.dataset import Dataset
from dsr.utils import MaxUniquePriorityQueue

from dsr.language_model.language_model import LModel


config_filename = 'config.json'
with open(config_filename, encoding='utf-8') as f:
    config = json.load(f)

config_dataset = config["dataset"]          # Problem specification hyperparameters
config_training = config["training"]        # Training hyperparameters
config_controller = config["controller"]    # Controller hyperparameters
config_lmodel = config["lmodel"]

# Define the dataset and library
dataset = Dataset(**config_dataset)
Program.set_training_data(dataset)
Program.set_library(dataset.function_set, dataset.n_input_var)
print("Ground truth expression:\n{}".format(indent(dataset.pretty(), '\t')))

with tf.Session() as sess:
    # Instantiate the controller
    lmodel = LModel(dataset.function_set, dataset.n_input_var, **config_lmodel)