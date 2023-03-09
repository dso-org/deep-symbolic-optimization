"""Generate model parity test case data for DeepSymbolicOptimizer."""

from pkg_resources import resource_filename

import tensorflow as tf
import click

from dso import DeepSymbolicOptimizer
from dso.config import load_config


# Shorter config run for parity test
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : 1000,
    "batch_size" : 100
}

@click.command()
@click.option('--stringency', '--t', default="strong", type=str, help="stringency of the test data to generate")
def main(stringency):
    # Load config
    config = load_config()

    # Train the model
    model = DeepSymbolicOptimizer(config)

    if stringency == "strong":
        n_samples = 1000
        suffix = "_" + stringency
    elif stringency == "weak":
        n_samples = 100
        suffix = "_" + stringency

    model.config_training.update({"n_samples" : n_samples,
                                  "batch_size" : 100})

    # Turn on GP meld
    model.config_gp_meld.update({"run_gp_meld" : True,
                                 "generations" : 3,
                                 "population_size" : 10})

    model.train()

    # Save the TF model
    tf_save_path = resource_filename("dso.test", "data/test_model" + suffix)
    saver = tf.train.Saver()
    saver.save(model.sess, tf_save_path)


if __name__ == "__main__":
    main()
