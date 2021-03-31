"""Sampling obs, and action data from a Zoo or dsr policy on a Gym environment.
Usage:
    - Run all envs for zoo and dsp: python latency.py
    - Run all envs for specific source: python latency.py --source zoo
    - Run specific env for zoo and dsp: python latency.py --env Pendulum-v0
    - Run specific env only for specific source: python latency.py --env Pendulum-v0 --source zoo
    - Change number of episodes: python latency.py --episodes 100
"""
import numpy as np
import click
import gym

import time

import dsr.task.control.utils as U
from dsr.program import Program, from_str_tokens
from dsr.task import set_task

REGRESSION_SEED_SHIFT = int(2e6)

ENVS = {
    #"BipedalWalker-v2": {},
    "CustomCartPoleContinuous-v0": {
        "n_actions" : 1,
        #"symbolic" : ["add,mul,10.0,x3,x4"]  # Ours
        "symbolic" : ["add,mul,31.9,x3,add,mul,8.2,x4,add,x1,mul,2.3,x2"] # LQR optimal
    },
    "HopperBulletEnv-v0" : {
        "n_actions" : 3,
        "env_kwargs" : {},
        "symbolic" : [
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4"]
    },
    #"InvertedDoublePendulumBulletEnv-v0" : {
    #    "n_actions" : 1,
    #    "symbolic" : ["1,0"]
    #},
    #"InvertedPendulumSwingupBulletEnv-v0" : {
    #    "n_actions" : 1,
    #    "symbolic" : ["1,0"]
    #},
    "LunarLanderContinuous-v2" : {
        "n_actions" : 2,
        "symbolic" : [
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4"]
    },
    "MountainCarContinuous-v0" : {
        "n_actions" : 1,
        "symbolic" : ["div,log,cos,1.0,log,x2"] # 99.09
    },
    "Pendulum-v0" : {
        "n_actions" : 1,
        "symbolic" : ["add,mul,-2.0,x2,div,add,mul,-8.0,x2,mul,-2.0,x3,x1"]
    },
    #"ReacherBulletEnv-v0" : {
    #    "n_actions" : 2,
    #    "symbolic" : ["1,0"]
    #}
}


class Model():
    def __init__(self, env_name, source="zoo"):
        self.source = source
        self.model = self.load_model(env_name)

    def load_model(self, env_name):
        if self.source == "zoo":
            U.load_default_model(env_name)
            return U.model
        elif self.source == "dsp":
            config_task = {
                "task_type" : "control",
                "name" : env_name,
                "anchor" : None,
                "algorithm" : None,
                "action_spec" : ENVS[env_name]["symbolic"],
                "n_episodes_test" : 1,
                "success_score" : 200.0,
                "function_set" : ["add","sub","mul","div","sin","cos","exp","log","const"],
                "protected" : False
            }
            # Generate the eval_function
            set_task(config_task)
            Program.clear_cache()
            action_models = []
            for traversal in ENVS[env_name]["symbolic"]:
                action_models.append(
                    from_str_tokens(traversal, optimize=False, skip_cache=False, n_objects=1))
            return action_models
        else:
            assert False, "Unknown source for model"

    def predict(self, obs):
        if self.source == "zoo":
            start_time = time.time()
            prediction = self.model.predict(obs)
            predict_duration = time.time() - start_time
        elif self.source == "dsp":
            actions = []
            predict_duration = 0
            for action_model in self.model:
                modified_state = np.expand_dims(obs, axis=0)
                start_time = time.time()
                predict = action_model.execute(modified_state)
                predict_duration += time.time() - start_time
                actions.append(predict[0])
            prediction = np.array(actions), None
        else:
            assert False, "Unknown source for model"
        return prediction, predict_duration

@click.command()
@click.option("--env", type=str, default=None, help="Name of environment to sample")
@click.option("--episodes", type=int, default=10, help="Number of episodes to sample.")
@click.option("--source", type=str, default=None, help="Source of model (zoo or dsp).")
def main(env=None,  episodes=10, source=None):
    env_names = {env: ENVS[env]} if isinstance(env, str) else ENVS
    sources = [source] if isinstance(source, str) else ["zoo", "dsp"]

    for source in sources:
        text = []
        for env_name in env_names:
            # Make gym environment
            env = gym.make(env_name)
            if "Bullet" in env_name:
                env = U.TimeFeatureWrapper(env)

            # Load model
            model_load_start = time.time()
            model = Model(env_name, source)
            model_load_duration = time.time() - model_load_start

            # Run episodes
            action_durations = []
            episode_rewards = []
            for i in range(episodes):
                env.seed(i + REGRESSION_SEED_SHIFT)
                obs = env.reset()
                done = False
                rewards = []
                while not done:
                    action_start_time = time.time()
                    [action, _states], predict_duration = model.predict(obs)
                    action_durations.append(predict_duration)
                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                episode_rewards.append(sum(rewards))
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            text.append("{} [action dim = {}]: {:.4f} ms [Model load time: {:.4f} s] [Avg. reward: {:.4f}]".format(
                env_name, action.shape, np.mean(action_durations)*1000., model_load_duration, avg_reward))
        # Print summary
        print(" ")
        print("=== {} =========================".format(source))
        print("Avg. action durations after {} episodes".format(episodes))
        for line in text:
            print(line)
    print("============================")

if __name__ == "__main__":
    main()
