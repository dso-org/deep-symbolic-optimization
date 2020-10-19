import gym
import dsr.task.control
import numpy as np

from argparse import ArgumentParser

def setup_envs_run_rollouts(seed: int = 0, rollout_length: int = 10):
    gym_discrete_env = gym.make("LunarLander-v2")
    gym_continuous_env = gym.make("LunarLanderContinuous-v2")

    custom_discrete_env = gym.make("LunarLanderCustom-v0", reward_shaping_coef=1, continuous=False)
    custom_continuous_env = gym.make("LunarLanderCustom-v0", reward_shaping_coef=1, continuous=True)

    gym_discrete_env.seed(seed)
    gym_continuous_env.seed(seed)

    custom_discrete_env.seed(seed)
    custom_continuous_env.seed(seed)

    discrete_rollouts = []

    print("Entering discrete environments rollouts...")
    # rollout discrete envs and record output
    obs = gym_discrete_env.reset()
    gym_r, gym_done = 0, False
    custom_obs = custom_discrete_env.reset()
    custom_r, custom_done = 0, False
    for i in range(rollout_length):
        action = gym_discrete_env.action_space.sample()

        this_step = ((obs, gym_r, gym_done), (custom_obs, custom_r, custom_done)) 
        discrete_rollouts.append(this_step)

        obs, gym_r, gym_done, _ = gym_discrete_env.step(action)
        custom_obs, custom_r, custom_done, _ = custom_discrete_env.step(action)

    last_step = ((obs, gym_r, gym_done), (custom_obs, custom_r, custom_done)) 
    discrete_rollouts.append(last_step)

    print("Discrete rollouts finished.")


    continuous_rollouts = []

    print("Entering continuous environments rollouts...")
    # rollout continuous envs and record output
    obs = gym_continuous_env.reset()
    gym_r, gym_done = 0, False
    custom_obs = custom_continuous_env.reset()
    custom_r, custom_done = 0, False
    for i in range(rollout_length):
        action = gym_continuous_env.action_space.sample()

        this_step = ((obs, gym_r, gym_done), (custom_obs, custom_r, custom_done)) 
        continuous_rollouts.append(this_step)

        obs, gym_r, gym_done, _ = gym_continuous_env.step(action)
        custom_obs, custom_r, custom_done, _ = custom_continuous_env.step(action)

    last_step = ((obs, gym_r, gym_done), (custom_obs, custom_r, custom_done)) 
    continuous_rollouts.append(last_step)

    print("Continuous rollouts finished.")

    print("Starting tests...")

    for step in discrete_rollouts:
        assert np.array_equal(step[0][0], step[1][0]), "States do not match."
        assert step[0][1] == step[1][1], "Rewards obtained do not match."
        assert step[0][2] == step[0][2], "Episode dones do not match."
    print("Discrete tests passed.")

    for step in continuous_rollouts:
        assert np.array_equal(step[0][0], step[1][0]), "States do not match."
        assert step[0][1] == step[1][1], "Rewards obtained do not match."
        assert step[0][2] == step[0][2], "Episode dones do not match."
    print("Continuous tests passed.")

    return 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-length', '--rollout_length', type=int, default=10)
    args = parser.parse_args()

    setup_envs_run_rollouts(args.seed, args.rollout_length)