from pkg_resources import resource_filename

import pytest
import numpy as np

from dsr import DeepSymbolicRegressor
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE


@pytest.fixture
def model():
    return DeepSymbolicRegressor()


@pytest.mark.parametrize("config", ["config.json"])
def test_task(model, config):
    """Test regression for various configs."""

    config = resource_filename("dsr", config)

    # Generate some data
    np.random.seed(0)
    X = np.random.random(size=(10, 3))
    y = np.random.random(size=(10,))

    model.update_config(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.fit(X, y)
