"""Sampling obs, and action data from a Zoo policy on a Gym environment. Do, python sample_zoo.py --env ENV_NAME """
import os
import sys

import numpy as np
import click
import gym

import dso.task.control.utils as U

DSP_DATA_ROOT = "." #../data"
DSO_DATA_ROOT = "./dso/task/regression/data" #"../../regression/data"
REGRESSION_SEED_SHIFT = int(2e6)


@click.command()
@click.option("--env", type=str, default="LunarLanderContinuous-v2", help="Name of environment to sample")
@click.option("--n_episodes", type=int, default=1000, help="Number of episodes to sample.")
@click.option("--n_samples", type=int, default=10000, help="Number of transitions to save.")
def main(env,  n_episodes, n_samples):

    env_name = env

    # Make gym environment
    env = gym.make(env_name)
    if "Bullet" in env_name:
       env = U.TimeFeatureWrapper(env)

    #Load model
    U.load_default_model(env_name)

    for i in range(n_episodes):
        env.seed(i + REGRESSION_SEED_SHIFT)
        obs=env.reset()
        done = False
        obs_list = []
        action_list = []
        while not done:
            obs_list.append(obs)
            action, _states = U.model.predict(obs)
            obs, rewards, done, info = env.step(action)
            action = np.array(action)
            action_list.append(action)

    # Convert to array
    # Columns correspond to [s1, s2, ..., sn, a1, a2, ..., an] for use by DSO
    obs_array = np.array(obs_list)
    action_array = np.array(action_list)
    data = np.hstack([obs_array, action_array])

    # Save all data as numpy file
    path = os.path.join(DSP_DATA_ROOT, env_name + ".npz")
    np.savez(path, data)

    # Save randomly subset as CSV
    np.random.seed(0) # For reproducibility
    rows_to_keep = np.random.choice(data.shape[0], n_samples, replace=False)
    data = data[rows_to_keep, :]
    path = os.path.join(DSO_DATA_ROOT, env_name + ".csv")
    np.savetxt(path, data, delimiter=",")


if __name__ == "__main__":
    main()


