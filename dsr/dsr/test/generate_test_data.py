"""Generate model parity test case data for DeepSymbolicOptimizer."""

from pkg_resources import resource_filename

from dsr import DeepSymbolicOptimizer
from dsr.config import load_config


# Shorter config run for parity test
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : 1000,
    "batch_size" : 100
}


def main():
    # Load config
    config = load_config()
    config["task"].pop("method")

    # Train the model
    model = DeepSymbolicOptimizer(config)
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    # Save the model
    save_path = resource_filename("dsr.test", "data/test_model")
    model.save(save_path)


if __name__ == "__main__":
    main()
