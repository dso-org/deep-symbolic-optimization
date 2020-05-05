"""Utility functions for control task."""

import os
from pkg_resources import resource_filename

try:
    import mpi4py
except ImportError:
    mpi4py = None

from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3

if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO


ALGORITHMS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}    


# NOTE: This does not load observation normalization
# Loads a model into global namespace
def load_model(algorithm, model_path):
    global model
    model = ALGORITHMS[algorithm].load(model_path)
    print("Loaded {} model {}".format(algorithm.upper(), model_path))


# Load an environment's default model, which is located at:
# dsr/task/control/data/[env_name]/model-[algorithm].[pkl | zip]
def load_default_model(env_name):

    # Find default algorithm and model path for the environment
    task_root = resource_filename("dsr.task", "control")
    root = os.path.join(task_root, "data", env_name)
    files = os.listdir(root)
    for f in files:
        if f.startswith("model-"):
            for ext in [".pkl", ".zip"]:
                if f.endswith(ext):
                    algorithm = f.split("model-")[-1].split(ext)[0]
                    model_path = os.path.join(root, f)
                    
                    # Load that model
                    load_model(algorithm, model_path)

                    return

    assert False, "Could not find default model for environment {}.".format(env_name)
