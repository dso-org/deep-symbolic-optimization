from gym import register

# Gym Pendulum-v0 with added dt parameter
register(
    id='CustomPendulum-v0',
    entry_point='dsr.task.control.envs.pendulum:CustomPendulumEnv',
    max_episode_steps=200,
)

# PyBullet CartPoleContinuousBulletEnv-v0 with added dt parameter
register(
    id='CustomCartPoleContinuousBulletEnv-v0',
    entry_point='dsr.task.control.envs.cartpole_bullet:CustomCartPoleContinuousBulletEnv',
    max_episode_steps=200,
    reward_threshold=190.0,
)

# Gym-modified ContinuousCartPole-v0 with added dt parameter
register(
    id='CustomContinuousCartPole-v0',
    entry_point='dsr.task.control.envs.continuous_cartpole:CustomContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)