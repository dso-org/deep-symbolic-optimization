"""Utility functions for control task."""

import os
from pkg_resources import resource_filename

from datetime import datetime
from glob import glob

import gym
from gym.wrappers import TimeLimit
from gym.wrappers.monitoring import video_recorder
import numpy as np

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
# dso/task/control/data/[env_name]/model-[algorithm].[pkl | zip]
def load_default_model(env_name):

    # Find default algorithm and model path for the environment
    task_root = resource_filename("dso.task", "control")
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


# From https://github.com/araffin/rl-baselines-zoo/blob/master/utils/wrappers.py
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        return self._get_obs(self.env.reset(**kwargs))

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class RenderEnv(gym.Wrapper):
    def __init__(self, env, env_name, alg, save_path):
        """Recording wrapper to simply generate videos from gym environments."""
        super(RenderEnv, self).__init__(env)

        self.env_name = env_name
        self.alg = alg

        self.videos = []
        self.video_recorder = None
        self.episode = 0

        self.directory = os.path.abspath(save_path)
        if not os.path.exists(self.directory): os.makedirs(self.directory, exist_ok=True)

    def reset(self, **kwargs):
        """Standard wrap of the reset function."""
        seed = kwargs.pop('seed', None)
        obs = self.env.reset(**kwargs)
        self.episode += 1
        self.reset_video_recorder(seed)
        return obs

    def step(self, action):
        """Standard wrap of the step function."""
        observation, reward, done, info = self.env.step(action)
        self.video_recorder.capture_frame()
        return observation, reward, done, info

    def reset_video_recorder(self, seed):
        """Close any existing video recorder and start recording the next video."""
        if self.video_recorder:
            self._close_video_recorder()

        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(
                self.directory,
                '{}.{}.s{}.{}'.format(self.env_name, self.alg, seed, datetime.now().strftime("%Y-%m-%d-%H%M%S"))),
            enabled=True,
        )
        self.video_recorder.capture_frame()

    def _close_video_recorder(self):
        """Cleaning up."""
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super(RenderEnv, self).close()
        if self.video_recorder is not None:
            self._close_video_recorder()
        self._clean_up_metadata()

    def __del__(self):
        """Make sure we've closed up shop when garbage collecting."""
        self.close()

    def _clean_up_metadata(self):
        """Deleting all metadata json files in the directory."""
        metadata_files = glob(os.path.join(self.directory , "*.json"))
        for metadata_file in metadata_files:
            os.remove(metadata_file)