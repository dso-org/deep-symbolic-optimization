"""Utility functions for control task."""

# Modules required to run dsp branch
import gym
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy


def load_anchor(model_path, env_name):
    global model
    env = gym.make(env_name) 
    model = DDPG(LnMlpPolicy, env, verbose=1)
    model = DDPG.load(model_path)
    print("Loaded model {}".format(model_path))
