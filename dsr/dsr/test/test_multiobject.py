import pytest
from pkg_resources import resource_filename

from dsr.core import DeepSymbolicOptimizer
from dsr.program import from_tokens, Program
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
import tensorflow as tf
import numpy as np

@pytest.fixture
def model():
    return DeepSymbolicOptimizer("../config_test_multiobject.json")


@pytest.fixture
def cached_results(model):
    save_path = resource_filename("dsr.test", "data/test_model")
    model.load(save_path)
    results = model.sess.run(tf.trainable_variables())

    return results

def test_model_parity(model, cached_results):
    """Compare results to last"""

    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    results = model.sess.run(tf.trainable_variables())

    cached_results = np.concatenate([a.flatten() for a in cached_results])
    results = np.concatenate([a.flatten() for a in results])
    np.testing.assert_array_almost_equal(results, cached_results)