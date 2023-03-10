import os

import pytest

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from datetime import datetime


N_STEPS = 3


def make_config(logdir=None, seed=None, timestamp=None):
    config = load_config()

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp
    config["experiment"]["logdir"] = str(logdir)

    # IMPORTANT: CHANGE THE SEED EACH CHECKPOINT
    # If the seed does not change each checkpoint, almost the same sequences
    # will be sampled again, even as neural network weights change.
    config["experiment"]["seed"] = seed

    # Turn on debugging to see neural network weights changing
    config["training"]["verbose"] = False
    config["training"]["debug"] = True

    return config


@pytest.mark.parametrize("pqt", [False, True])
def test_checkpoint_manual(tmp_path, pqt):
    """
    Run a model N_STEPS iterations, manually saving to/loading from
    checkpointing each iteration with a new model instance.
    """

    timestamp = None
    for i in range(N_STEPS):
        load_path = os.path.join(tmp_path, "checkpoint_{}".format(i - 1)) if i > 0 else None
        save_path = os.path.join(tmp_path, "checkpoint_{}".format(i))

        config = make_config(logdir=tmp_path, seed=i, timestamp=timestamp)
        if pqt:
            config["policy_optimizer"] = {
                "policy_optimizer_type" : "pqt",
                "pqt_k" : 10,
                "pqt_batch_size" : 3
            }
        timestamp = config["experiment"]["timestamp"] # Reuse for next checkpoint

        # Load the model from checkpoint, train it one step, then save the checkpoint
        model = DeepSymbolicOptimizer(config)
        model.setup()
        if load_path is not None:
            model.load(load_path)
            if pqt:
                assert len(model.trainer.priority_queue) == i
        model.train_one_step()
        model.save(save_path)


def test_checkpoint_config(tmp_path):
    """
    Run a model N_STEPS iterations, checkpointing via config.
    """

    config = make_config(logdir=tmp_path, seed=0, timestamp=None)
    config["checkpoint"] = {
        "save_freq" : 1,
        "units" : "iterations"
    }

    model = DeepSymbolicOptimizer(config)
    for i in range(N_STEPS):
        model.train_one_step()
