import json
import pytest

import sys
sys.path.append('../')
from dsr import DeepSymbolicOptimizer
import numpy as np


# @pytest.fixture
# def model():
#     return DeepSymbolicOptimizer("./test/binding_task/data/config.json")

@pytest.mark.parametrize("config_file", ['./test/binding_task/data/config_full.json',
                                         './test/binding_task/data/config_full.json'])
def test_task_execution(config_file):
    model = DeepSymbolicOptimizer(config_file)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.config_prior["seq_positions"]["yaml_file"] = './test/binding_task/data/positions_prior.yaml'
    model.config_task['paths']['use_gpu'] = False
    model.train()


if __name__ == '__main__':

    config = json.load(open("./test/binding_task/data/config_short.json", 'rb'))
    m = DeepSymbolicOptimizer(config)
    m.train()