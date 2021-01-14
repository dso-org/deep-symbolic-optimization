"""Test cases for DeepSymbolicOptimizer on each Task."""

from pkg_resources import resource_filename

import pytest
import tensorflow as tf
import numpy as np

from dsr import DeepSymbolicOptimizer
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE


@pytest.fixture
def model():
    return DeepSymbolicOptimizer("config.json")


@pytest.fixture
def cached_results(model):
    save_path = resource_filename("dsr.test", "data/test_model")
    model.load(save_path)
    results = model.sess.run(tf.trainable_variables())

    return results


@pytest.mark.parametrize("config", ["config.json"])
def test_task(model, config):
    """Test that Tasks do not crash for various configs."""

    model.update_config(config)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.train()


def test_model_parity(model, cached_results):
    """Compare results to last"""

    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    results = model.sess.run(tf.trainable_variables())

    cached_results = np.concatenate([a.flatten() for a in cached_results])
    results = np.concatenate([a.flatten() for a in results])
    np.testing.assert_array_almost_equal(results, cached_results)
