import gym
import dsr.task.control
import numpy as np

from argparse import ArgumentParser

def setup_envs_run_rollouts(seed: int = 0, rollout_length: int = 10):
    continuous_envs = {"LunarLanderContinuous-v2": {}, "LunarLanderCustom-v0": dict(reward_shaping_coef=1, continuous=True)}
    discrete_envs = {"LunarLander-v2": {}, "LunarLanderCustom-v0": dict(reward_shaping_coef=1, continuous=False)}
    all_envs = [discrete_envs, continuous_envs]

    results = {}
    for i, envs in enumerate(all_envs):
        for env, kwargs in envs.items():
            current_env = gym.make(env, **kwargs)
            current_env.seed(seed)
            current_env.action_space.seed(seed)
            current_env.observation_space.seed(seed)

            results[env] = {}

            obs = current_env.reset()
            r, done = 0, False
            for i in range(rollout_length):
                action = current_env.action_space.sample()

                results[env]["state"] = obs
                results[env]["reward"] = r
                results[env]["done"] = done

                obs, r, done, _ = current_env.step(action)

            results[env]["state"] = obs
            results[env]["reward"] = r
            results[env]["done"] = done

        names = list(envs.keys())
        print(f"Comparing: {names}...")
        assert np.array_equal(results[names[0]]['state'], results[names[1]]['state']), "States do not match between above environments."
        assert np.array_equal(results[names[0]]['reward'], results[names[1]]['reward']), "Rewards do not match between above environments."
        assert np.array_equal(results[names[0]]['done'], results[names[1]]['done']), "Episode dones do not match between above environments."
        print(f"Tests passed.")
    
    return 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-length', '--rollout_length', type=int, default=10)
    args = parser.parse_args()

    setup_envs_run_rollouts(args.seed, args.rollout_length)