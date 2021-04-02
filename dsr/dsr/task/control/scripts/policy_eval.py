"""Sampling obs, and action data from a Zoo or dsr policy on a Gym environment.
Usage:
    - Run all envs for zoo and dsp: python latency.py
    - Run all envs for specific source: python latency.py --source zoo
    - Run specific env for zoo and dsp: python latency.py --env Pendulum-v0
    - Run specific env only for specific source: python latency.py --env Pendulum-v0 --source zoo
    - Change number of episodes: python latency.py --episodes 100
For debugging we can shorten running time and print more information:
    - Print env information: python latency.py --env Pendulum-v0 --print_env
    - Print action/state/reward per step: python latency.py --env Pendulum-v0 --print_action --print_state --print_reward
    - Same as above: python latency.py --env Pendulum-v0 --print_all
"""
import csv
import os
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
        "symbolic" : ["add,mul,10.0,x3,x4"]  # Ours
        #"symbolic" : ["add,mul,31.9,x3,add,mul,8.2,x4,add,x1,mul,2.3,x2"] # LQR optimal
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


def get_env_info(env_name, env):
    print(" ")
    print("==========================================")
    print("Env: {}".format(env_name))
    print("Action space: {} --> Single Agent Sample: {}".format(env.action_space, env.action_space.sample()))
    print("Observation space: {} --> Single Agent Sample: {}".format(env.observation_space, env.reset()))
    print("==========================================")


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
@click.option("--max_steps", type=int, default=None, help="Max number of steps per episodes.")
@click.option("--source", type=str, default=None, help="Source of model (zoo or dsp).")
@click.option("--seed", type=int, default=0, help="Environment seed.")
@click.option("--print_env", is_flag=True, help="Print out information about the environment.")
@click.option("--print_state", is_flag=True, help="Simple way to observe states when stepping through an environment.")
@click.option("--print_action", is_flag=True, help="Simple way to observe actions when stepping through an environment.")
@click.option("--print_reward", is_flag=True, help="Simple way to observe rewards when stepping through an environment.")
@click.option("--print_all", is_flag=True, help="Simple way to observe everything when stepping through an environment.")
def main(env=None,  episodes=10, max_steps=None, source=None, seed=0,
        print_env=False, print_state=False, print_action=False, print_reward=False, print_all=False):
    env_names = {env: ENVS[env]} if isinstance(env, str) else ENVS
    sources = [source] if isinstance(source, str) else ["zoo", "dsp"]

    if print_all:
        print_env = True
        print_state = True
        print_action = True
        print_reward = True

    for source in sources:
        csv_content= []
        text = []
        for env_name in env_names:
            # Make gym environment
            env = gym.make(env_name)
            if "Bullet" in env_name:
                env = U.TimeFeatureWrapper(env)
            if print_env:
                get_env_info(env_name, env)

            if max_steps is None:
                max_steps = env._max_episode_steps

            # Load model
            model_load_start = time.time()
            model = Model(env_name, source)
            model_load_duration = time.time() - model_load_start

            # Run episodes
            action_durations = []
            episode_rewards = []
            episode_steps = []
            for i in range(episodes):
                episode_step = 1
                env.seed(seed + i + REGRESSION_SEED_SHIFT)
                obs = env.reset()
                if print_state:
                    print("[E {:3d}/S {:3d}] S:".format(i + 1, episode_step - 1), ["{:.4f}".format(x) for x in obs])
                done = False
                rewards = []
                while not done:
                    [action, _states], predict_duration = model.predict(obs)
                    if print_action:
                        print("[E {:3d}/S {:3d}] A:".format(i + 1, episode_step), ["{:.4f}".format(x) for x in action])
                    action_durations.append(predict_duration)
                    obs, reward, done, info = env.step(action)
                    if print_reward:
                        print("[E {:3d}/S {:3d}] R: {:.4f}".format(i + 1, episode_step, reward))
                    if print_state:
                        print("[E {:3d}/S {:3d}] S:".format(i + 1, episode_step), ["{:.4f}".format(x) for x in obs])
                    rewards.append(reward)
                    episode_step += 1
                    if not done and max_steps == episode_step:
                        done = True
                episode_rewards.append(sum(rewards))
                episode_steps.append(episode_step)
            text.append("{} [action dim = {}] --> [Reward: {:.4f}] [Steps: {:3d}] [Action latency: {:.4f} ms] [Model load time: {:.4f} s]".format(
                env_name, action.shape, np.mean(episode_rewards), int(np.mean(episode_steps)), np.mean(action_durations)*1000., model_load_duration))
            csv_content.append([env_name, max_steps, episodes, int(np.mean(episode_steps)),
                np.mean(episode_rewards),model_load_duration,np.mean(action_durations)*1000.])
        # Print summary
        print("=== {} === Averages over {} episodes =========================".format(source, episodes))
        for line in text:
            print(line)
        file_name = 'policy_eval_results_{}.csv'.format(source)
        if not os.path.exists(file_name):
            with open(file_name,'w') as result_file:
                file_pointer = csv.writer(result_file, dialect='excel')
                file_pointer.writerow(
                    ["environment", "max_steps", "episodes", "avg_steps_episode",
                     "avg_rewards_episode", "model_load_s", "action_latency_ms"])
        with open(file_name,'a') as result_file:
            file_pointer = csv.writer(result_file, dialect='excel')
            file_pointer.writerows(csv_content)
    print("============================")

if __name__ == "__main__":
    main()
