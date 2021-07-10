"""Generate model parity test case data for DeepSymbolicOptimizer."""

from pkg_resources import resource_filename

from dso import DeepSymbolicOptimizer
from dso.config import load_config


# Shorter config run for parity test
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : 1000,
    "batch_size" : 100
}


def main():
    # Load config
    config = load_config()

    # Train the model
    model = DeepSymbolicOptimizer(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    # Save the model
    save_path = resource_filename("dso.test", "data/test_model")
    model.save(save_path)


if __name__ == "__main__":
    main()
