import pytest
import numpy as np

from dsr import DeepSymbolicRegressor


SEED = 0
N_SAMPLES = 10
BATCH_SIZE = 5
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : N_SAMPLES,
    "batch_size" : BATCH_SIZE
}


@pytest.fixture
def model():
    return DeepSymbolicRegressor()


@pytest.mark.parametrize("config", ["../../config.json"])
def test_task(model, config):
    """Test regression for various configs."""

    # Generate some data
    np.random.seed(0)
    X = np.random.random(size=(10, 3))
    y = np.random.random(size=(10,))

    model.update_config(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.fit(X, y)
