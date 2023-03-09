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


@pytest.fixture(params=("strong", "weak"))
def cached_results(model, request):
    if request.param == "strong":
        model_data = "data/test_model_strong"
    elif request.param == "weak":
        model_data = "data/test_model_weak"
    tf_load_path = resource_filename("dso.test", model_data)
    model.setup()
    saver = tf.train.Saver()
    saver.restore(model.sess, tf_load_path)
    results = model.sess.run(tf.trainable_variables())

    return [request.param, results]


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
    """Compare results with gp meld on to previous set"""
    config = load_config(config)
    config["experiment"]["logdir"] = None # Turn off saving results
    model.set_config(config)

    [stringency, cached_results]= cached_results

    if stringency == "strong":
        n_samples = 1000
    elif stringency == "weak":
        n_samples = 100

    model.config_training.update({"n_samples" : n_samples,
                                  "batch_size" : 100})

    # Turn on GP meld
    model.config_gp_meld.update({"run_gp_meld" : True,
                                 "generations" : 3,
                                 "population_size" : 10,
                                 "crossover_operator" : "cxOnePoint",
                                 "mutation_operator" : "multi_mutate"
                                 })

    model.train()
    results = model.sess.run(tf.trainable_variables())
    results = np.concatenate([a.flatten() for a in results])
    cached_results = np.concatenate([a.flatten() for a in cached_results])
    assert np.linalg.norm(cached_results, ord=1) > 0
    
    if stringency == "weak":
        results = np.where(results, 1, 0)
        cached_results = np.where(cached_results, 1, 0)

    np.testing.assert_array_almost_equal(results, cached_results)
