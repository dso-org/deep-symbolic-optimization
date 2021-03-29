import gym
import numpy as np
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
print(pybullet_envs)
env = gym.make("LunarLanderContinuous-v2")
env.seed(0)
env.reset()
action = env.action_space.sample() # your agent here (this takes random actions)
observation, reward, done, info = env.step(action)
dimensionAction = len(action)
print(dimensionAction)
actionfix =  np.random.randn(dimensionAction,)
numberIterations = 10

env.seed(0)
env.reset()
numberIterations = 10
for i in range(numberIterations):
    obs = env.step(actionfix)[0]
    print("AFTER ACTION:", ["{:.4f}".format(x) for x in obs])