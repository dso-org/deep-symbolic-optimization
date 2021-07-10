"""Test that given the same seed, changing dt does not alter the starting state and does alter the next state"""

import gym
import dso.task.control.envs # Needed to register the environments so they're seen by gym.make()
import numpy as np

env_ids = ["CustomPendulum-v0", "CustomCartPoleContinuous-v0", "CustomCartPoleContinuousBulletEnv-v0"]
dts = [0.01, 0.02]

data = {}

for env_id in env_ids:

    data[env_id] = {}

    for dt in dts:

        data[env_id][dt] = {}

        env = gym.make(env_id, dt=dt)
        env.seed(0)
        s0 = env.reset()
        a = env.action_space.sample()
        s1 = env.step(a)[0]

        data[env_id][dt]["s0"] = s0
        data[env_id][dt]["s1"] = s1

    # Assert starting states are identical
    s0 = data[env_id][dts[0]]["s0"]
    for dt in dts[1:]:
        assert np.array_equal(s0, data[env_id][dt]["s0"]), "Starting states for {} are not identical.".format(env_id)

    # Assert next states are not identical
    s1 = data[env_id][dts[0]]["s1"]
    for dt in dts[1:]:
        assert not np.array_equal(s1, data[env_id][dt]["s1"]), "Next states for {} are identical.".format(env_id)

    print("Test passed for {}.".format(env_id))

print("All tests passed.")