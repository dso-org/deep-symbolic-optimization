"""Sampling obs, and action data from a Zoo or DSO policy on a Gym environment.
Usage:
    - Run all envs for zoo and symbolic: python policy_eval.py
    - Run all envs for specific alg: python policy_eval.py --alg zoo
    - Run specific env for zoo and symbolic: python policy_eval.py --env Pendulum-v0
    - Run specific env only for specific alg: python policy_eval.py --env Pendulum-v0 --alg zoo
    - Change number of episodes: python policy_eval.py --episodes 100
For debugging we can shorten running time and print more information:
    - Print env information: python policy_eval.py --env Pendulum-v0 --print_env
    - Print action/state/reward per step: python policy_eval.py --env Pendulum-v0 --print_action --print_state --print_reward
    - Same as above: python policy_eval.py --env Pendulum-v0 --print_all
"""
import csv
import glob
import os
import numpy as np
import click
import gym
import subprocess

from datetime import datetime
import time

from pkg_resources import resource_filename

import dso.task.control.utils as U
from dso.program import Program, from_str_tokens
from dso.task import set_task

REGRESSION_SEED_SHIFT = int(2e6)
DEFAULT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
POSITIVE_SCALES = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ENVS = {
    "CustomCartPoleContinuous-v0": {
        "n_actions" : 1,
        "env_kwargs" : {
            "dt_multiplier" : DEFAULT_SCALES
        },
        "symbolic" : ["add,mul,10.0,x3,x4"],  # Ours
        #"symbolic" : ["add,mul,31.9,x3,add,mul,8.2,x4,add,x1,mul,2.3,x2"], # LQR optimal
    },
    "HopperBulletEnv-v0" : {
        "n_actions" : 3,
        "env_kwargs" : {},
        "symbolic" : [
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4"],
    },
    #"InvertedDoublePendulumBulletEnv-v0" : {
    #    "n_actions" : 1,
    #    "symbolic" : [],
    #},
    "InvertedPendulumSwingupBulletEnv-v0" : {
        "n_actions" : 1,
        "symbolic" : [],
    },
    "LunarLanderContinuous-v2" : {
        "n_actions" : 2,
        "symbolic" : [
            'add,add,mul,0.7624715039886016,sub,sin,x3,add,add,add,add,div,add,sin,add,0.09101695879983959,0.0922444066126055,' \
                'add,x2,x4,0.11289444161591844,x4,x4,x4,x4,-0.0008359813574848967,-0.12271695008045375',
            'add,add,mul,0.24326721693967257,div,div,x4,log,sin,1.019536868443063,sub,x6,x3,-0.07620830192712105,0.07787010100098943'
            ],
    },
    "MountainCarContinuous-v0" : {
        "n_actions" : 1,
        "env_kwargs" : {
            "power_multiplier" : DEFAULT_SCALES,
            "starting_state_multiplier" : POSITIVE_SCALES
        },
        "symbolic" : ["div,log,cos,1.0,log,x2"], # 99.09
        # "symbolic" : ["div,mul,0.05,log,x2,add,x2,log,mul,10.0,x2"],
    },
    "Pendulum-v0" : {
        "n_actions" : 1,
        "env_kwargs" : {
            "dt_multiplier" : DEFAULT_SCALES,
            # "gravity_multiplier" : DEFAULT_SCALES,
            # "mass_multiplier" : DEFAULT_SCALES,
            # "length_multiplier" : DEFAULT_SCALES,
            # "starting_state_multiplier" : POSITIVE_SCALES
        },
        "symbolic" : ["add,mul,-2.0,x2,div,add,mul,-8.0,x2,mul,-2.0,x3,x1"],
    },
    "ReacherBulletEnv-v0" : {
        "n_actions" : 2,
        "symbolic" : [],
    }
}

ALGS = [
    'a2c',
    'acktr',
    'ddpg',
    'sac',
    'ppo2',
    'trpo',
    'td3',
    'symbolic'
]

def get_env_info(env_name, env):
    print(" ")
    print("==========================================")
    print("Env: {}".format(env_name))
    print("Action space: {} --> Single Agent Sample: {}".format(env.action_space, env.action_space.sample()))
    print("Observation space: {} --> Single Agent Sample: {}".format(env.observation_space, env.reset()))
    print("==========================================")


class Model():
    def __init__(self, env_name, alg="zoo"):
        self.alg = alg
        self.model = self.load_model(env_name)

    def load_model(self, env_name):
        if self.alg != "symbolic":
            env_models_path = os.path.join(resource_filename("dso.task", "control"), "data", env_name)
            alg_models = glob.glob("{}/*-{}.*".format(env_models_path, self.alg.lower()))
            if len(alg_models) == 0:
                print("WARNING: No '{}' model available for '{}'!".format(self.alg.upper(), env_name))
                return None
            try:
                U.load_model(self.alg, alg_models[0])
            except:
                print("WARNING: Model available but could not load: '{}'".format(alg_models[0]))
                return None
            return U.model
        else:
            if len(ENVS[env_name]["symbolic"]) == 0:
                 print("WARNING: No symbolic policy available for env '{}'!".format(env_name))
                 return None
            config_task = {
                "task_type" : "control",
                "env" : env_name,
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
                    from_str_tokens(traversal, skip_cache=False, n_objects=1))
            return action_models

    def predict(self, obs):
        if self.alg != "symbolic":
            start_time = time.time()
            prediction = self.model.predict(obs)
            predict_duration = time.time() - start_time
        else:
            actions = []
            predict_duration = 0
            for action_model in self.model:
                modified_state = np.expand_dims(obs, axis=0)
                start_time = time.time()
                predict = action_model.execute(modified_state)
                predict_duration += time.time() - start_time
                actions.append(predict[0])
            prediction = np.array(actions), None
        return prediction, predict_duration

@click.command()
@click.option("--env", multiple=True, type=str, help="Name of environment to run (default: all)")
@click.option("--alg", multiple=True, type=str, help="Algorithm to run (default: all).")
@click.option("--episodes", type=int, default=10, help="Number of episodes to sample.")
@click.option("--max_steps", type=int, default=None, help="Max number of steps per episodes.")
@click.option("--seed", type=int, default=0, help="Environment seed.")
@click.option("--print_env", is_flag=True, help="Print out information about the environment.")
@click.option("--print_state", is_flag=True, help="Simple way to observe states when stepping through an environment.")
@click.option("--print_action", is_flag=True, help="Simple way to observe actions when stepping through an environment.")
@click.option("--print_reward", is_flag=True, help="Simple way to observe rewards when stepping through an environment.")
@click.option("--print_all", is_flag=True, help="Simple way to observe everything when stepping through an environment.")
@click.option("--record", is_flag=True, help="Record the policy in the environment to an mp4.")
def main(env, alg, episodes, max_steps, seed=0,
        print_env=False, print_state=False, print_action=False, print_reward=False, print_all=False, record=False):

    # Get commit label
    commit_label = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    # Preparing envs from input
    if all([isinstance(e, str) for e in env]) and all([e != "all" for e in env]) and len(env) > 0:
        #assert all([e in ENVS for e in env]), "ERROR: Environment '{}' unknown!".format(env)
        if not all([e in ENVS for e in env]):
            print("WARNING: {} environment not available!".format([e for e in env if e not in ENVS]))
        exp_envs = {e: ENVS[e] for e in env if e in ENVS}
    else:
        exp_envs = ENVS

    # Preparing algs from input
    if all([isinstance(a, str) for a in alg]) and all([a != "all" for a in alg]) and len(alg) > 0:
        if not all([a.lower() in ALGS for a in alg]):
            print("WARNING: {} algorithm not available!".format([a for a in alg if a.lower() not in ALGS]))
        exp_algs = [a.lower() for a in alg if a.lower() in ALGS]
    else:
        exp_algs = ALGS

    # Update output if necessary
    if print_all:
        print_env = True
        print_state = True
        print_action = True
        print_reward = True

    # Run experiments for each algorithm and environment combination
    for alg in exp_algs:
        csv_content= []
        text = []
        for env_name in exp_envs:
            # Prepare the gym environment
            env = gym.make(env_name)
            if "Bullet" in env_name:
                env = U.TimeFeatureWrapper(env)
            if print_env:
                get_env_info(env_name, env)
            if record:
                save_path = "videos/{}".format(env_name)
                env = U.RenderEnv(env, env_name, alg, save_path)
                text.append("Saving videos to: {}".format(save_path))

            # Load model
            model_load_start = time.time()
            model = Model(env_name, alg)
            model_load_duration = time.time() - model_load_start

            if model.model is None:
                continue

            # Run episodes
            action_durations = []
            episode_rewards = []
            episode_steps = []
            for i in range(episodes):
                episode_step = 1
                generated_seed = env.seed(seed + i + REGRESSION_SEED_SHIFT)
                obs = env.reset(seed=generated_seed[0])
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
                    if max_steps == episode_step:
                        done = True
                episode_rewards.append(sum(rewards))
                episode_steps.append(episode_step)
            text.append("{} [action dim = {}] --> [Reward: {:.4f}] [Steps: {:3d}] [Action latency: {:.4f} ms] [Model load time: {:.4f} s]".format(
                env_name, action.shape, np.mean(episode_rewards), int(np.mean(episode_steps)), np.mean(action_durations)*1000., model_load_duration))
            csv_content.append([env_name, alg, episodes, int(np.mean(episode_steps)),
                np.mean(episode_rewards), model_load_duration, np.mean(action_durations)*1000.,
                datetime.now(), commit_label])

        # Print summary
        print("=== {} === Averages over {} episodes =========================".format(alg, episodes))
        if len(text) > 0:
            for line in text:
                print(line)
            file_name = 'policy_eval_results.csv'
            if not os.path.exists(file_name):
                with open(file_name,'w') as result_file:
                    file_pointer = csv.writer(result_file, dialect='excel')
                    file_pointer.writerow(
                        ["environment", "algorithm", "episodes", "avg_steps_episode",
                        "avg_rewards_episode", "model_load_s", "action_latency_ms", "date", "commit"])
            with open(file_name,'a') as result_file:
                file_pointer = csv.writer(result_file, dialect='excel')
                file_pointer.writerows(csv_content)
        else:
            print("No algorithm/environment combinations found.")
        print("")
    print("============================")

if __name__ == "__main__":
    main()
