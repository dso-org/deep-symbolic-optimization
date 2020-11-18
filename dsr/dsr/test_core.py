"""Test cases for DeepSymbolicOptimizer on each Task."""

import pytest
import tensorflow as tf

from dsr import DeepSymbolicOptimizer


SEED = 0
N_SAMPLES = 10
BATCH_SIZE = 5
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : N_SAMPLES,
    "batch_size" : BATCH_SIZE
}


@pytest.fixture(autouse=True)
def setup_tf():
    """Reset the tensorflow graph so it can be reused across tests."""
    tf.set_random_seed(SEED)
    yield
    tf.reset_default_graph()


@pytest.fixture
def model():
    return DeepSymbolicOptimizer()


@pytest.mark.parametrize("config", ["config.json", "config_dsp.json"])
def test_task(model, config):
    """Test Tasks for various configs."""

    model.update_config(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
