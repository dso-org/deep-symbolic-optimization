from gym import register
import pybullet_envs
# the above import is here to register the pybullet environments to Gym. Don't delete!
# without the import you won't be able to use pybullet environments.

# Gym Pendulum-v0 with added dt parameter
register(
    id='CustomPendulum-v0',
    entry_point='dso.task.control.envs.pendulum:CustomPendulumEnv',
    max_episode_steps=200,
)

# PyBullet CartPoleContinuousBulletEnv-v0 with added dt parameter
register(
    id='CustomCartPoleContinuousBulletEnv-v0',
    entry_point='dso.task.control.envs.cartpole_bullet:CustomCartPoleContinuousBulletEnv',
    max_episode_steps=200,
    reward_threshold=190.0,
)

# Gym-modified ContinuousCartPole-v0 with added dt parameter
register(
    id='CustomCartPoleContinuous-v0',
    entry_point='dso.task.control.envs.continuous_cartpole:CustomCartPoleContinuousEnv',
    max_episode_steps=1000,
    reward_threshold=995.0,
)

# Modified discrete LunarLander to turn off reward shaping.
register(
    id='LunarLanderNoRewardShaping-v0',
    entry_point='dso.task.control.envs.lander:CustomLunarLander',
    kwargs=dict(reward_shaping_coef=0, continuous=False),
    max_episode_steps=1000,
    reward_threshold=200,
)

# Modified continuous LunarLander to turn off reward shaping.
register(
    id='LunarLanderContinuousNoRewardShaping-v0',
    entry_point='dso.task.control.envs.lander:CustomLunarLander',
    kwargs=dict(reward_shaping_coef=0, continuous=True),
    max_episode_steps=1000,
    reward_threshold=200,
)

# Modified LunarLander with no kwargs provided, it is up to the user to declare kwargs.
register(
    id='LunarLanderCustom-v0',
    entry_point='dso.task.control.envs.lander:CustomLunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)