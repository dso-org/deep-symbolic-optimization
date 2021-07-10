import gym
import dso.task.control
import numpy as np

import click

@click.command()
@click.option('--seed', default=0, help='Value for seeding environments.')
@click.option('--rollout_length', default=10, help='How long of episodes to do rollouts for comparing environments.')
def setup_envs_run_rollouts(seed: int, rollout_length: int):
    continuous_envs = {"LunarLanderContinuous-v2": {}, "LunarLanderCustom-v0": dict(reward_shaping_coef=1, continuous=True)}
    discrete_envs = {"LunarLander-v2": {}, "LunarLanderCustom-v0": dict(reward_shaping_coef=1, continuous=False)}
    all_envs = [discrete_envs, continuous_envs]

    results = {}
    for i, envs in enumerate(all_envs):
        for env_name, kwargs in envs.items():
            env = gym.make(env_name, **kwargs)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            results[env_name] = {}

            obs = env.reset()
            r, done = 0, False
            for i in range(rollout_length):
                action = env.action_space.sample()

                results[env_name]["state"] = obs
                results[env_name]["reward"] = r
                results[env_name]["done"] = done

                obs, r, done, _ = env.step(action)

            results[env_name]["state"] = obs
            results[env_name]["reward"] = r
            results[env_name]["done"] = done

        names = list(envs.keys())
        print(f"Comparing: {names}...")
        assert np.array_equal(results[names[0]]['state'], results[names[1]]['state']), "States do not match between above environments."
        assert np.array_equal(results[names[0]]['reward'], results[names[1]]['reward']), "Rewards do not match between above environments."
        assert np.array_equal(results[names[0]]['done'], results[names[1]]['done']), "Episode dones do not match between above environments."
        print(f"Tests passed.")
    
    return 0

if __name__ == '__main__':
    setup_envs_run_rollouts()