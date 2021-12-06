"""Test cases for DeepSymbolicOptimizer on each Task."""

from pkg_resources import resource_filename

import pytest
import tensorflow as tf
import numpy as np

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from dso.test.generate_test_data import CONFIG_TRAINING_OVERRIDE


@pytest.fixture
def model():
    config = load_config()
    config["experiment"]["logdir"] = None # Turn off saving results
    return DeepSymbolicOptimizer(config)


@pytest.fixture
def cached_results(model):
    save_path = resource_filename("dso.test", "data/test_model")
    model.load(save_path)
    results = model.sess.run(tf.trainable_variables())

    return results


@pytest.mark.parametrize("config", ["config/config_regression.json",
                                    "config/config_control.json"])

def test_task(model, config):
    """Test that Tasks do not crash for various configs."""
    config = load_config(config)
    config["experiment"]["logdir"] = None # Turn off saving results
    model.set_config(config)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.train()


@pytest.mark.parametrize("config", ["config/config_regression.json"])
def test_model_parity(model, cached_results, config):
    """Compare results to last"""

    config = load_config(config)
    config["experiment"]["logdir"] = None # Turn off saving results
    model.set_config(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    results = model.sess.run(tf.trainable_variables())

    cached_results = np.concatenate([a.flatten() for a in cached_results])
    results = np.concatenate([a.flatten() for a in results])
    np.testing.assert_array_almost_equal(results, cached_results)
