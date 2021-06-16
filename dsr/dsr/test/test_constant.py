import pytest

import numpy as np

from dsr.library import HardCodedConstant
from dsr.config import load_config
from dsr.core import DeepSymbolicOptimizer

@pytest.fixture()
def model():
    config = load_config()
    return DeepSymbolicOptimizer(config)

def test_constant():
    valid_cases = np.arange(0, 25, 0.1)
    for number in valid_cases:
        const = HardCodedConstant(value=number)
        assert const() == number, "Value returned from Constant.function() ({}) does not match input value ({}).".format(const(), number)

def test_regression_with_hard_coded_constants(model):
    model.config["task"]["function_set"] = ["add", "sin", 0, -1.0, 25, "1.23"]
    model.config_training.update(
        {
            "n_samples" : 10,
            "batch_size" : 4
        }
    )
    model.train()