"""Sampling obs, and action data from a Zoo or dsr policy on a Gym environment.
Usage:
    - Run all envs for zoo and dsp: python policy_eval.py
    - Run all envs for specific alg: python policy_eval.py --alg zoo
    - Run specific env for zoo and dsp: python policy_eval.py --env Pendulum-v0
    - Run specific env only for specific alg: python policy_eval.py --env Pendulum-v0 --alg zoo
    - Change number of episodes: python policy_eval.py --episodes 100
For debugging we can shorten running time and print more information:
    - Print env information: python policy_eval.py --env Pendulum-v0 --print_env
    - Print action/state/reward per step: python policy_eval.py --env Pendulum-v0 --print_action --print_state --print_reward
    - Same as above: python policy_eval.py --env Pendulum-v0 --print_all
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
DEFAULT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
POSITIVE_SCALES = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
ENVS = {
    #"BipedalWalker-v2": {},
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
    #    "symbolic" : ["1,0"],
    #},
    #"InvertedPendulumSwingupBulletEnv-v0" : {
    #    "n_actions" : 1,
    #    "symbolic" : ["1,0"],
    #},
    "LunarLanderContinuous-v2" : {
        "n_actions" : 2,
        "symbolic" : [
            "add,mul,10.0,x3,x4",
            "add,mul,10.0,x3,x4"],
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
    #"ReacherBulletEnv-v0" : {
    #    "n_actions" : 2,
    #    "symbolic" : ["1,0"],
    #}
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
        if self.alg == "zoo":
            U.load_default_model(env_name)
            return U.model
        elif self.alg == "dsp":
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
            assert False, "Unknown alg for model"

    def predict(self, obs):
        if self.alg == "zoo":
            start_time = time.time()
            prediction = self.model.predict(obs)
            predict_duration = time.time() - start_time
        elif self.alg == "dsp":
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
            assert False, "Unknown alg for model"
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
def main(env,  episodes, max_steps, alg, seed=0,
        print_env=False, print_state=False, print_action=False, print_reward=False, print_all=False):

    # Preparing envs from input
    if all([isinstance(e, str) for e in env]) and all([e != "all" for e in env]):
        assert all([e in ENVS for e in env]), "ERROR: Environment '{}' unknown!".format(env)
        exp_envs = {e: ENVS[e] for e in env}
    else:
        exp_envs = ENVS

    # Preparing algs from input
    if isinstance(alg, str) and alg != "all":
        assert alg in ALGS, "ERROR: Algorithm '{}' unknown!".format(alg.upper())
        exp_algs = {alg: ALGS[alg]}
    else:
        exp_algs = ALGS
    exp_algs = [alg] if isinstance(alg, str) else ["zoo", "dsp"]

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
            # Make gym environment
            env = gym.make(env_name)
            if "Bullet" in env_name:
                env = U.TimeFeatureWrapper(env)
            if print_env:
                get_env_info(env_name, env)

            if max_steps is None and '_max_episode_steps' in dir(env):
                max_steps = env._max_episode_steps

            # Load model
            model_load_start = time.time()
            model = Model(env_name, alg)
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
        print("=== {} === Averages over {} episodes =========================".format(alg, episodes))
        for line in text:
            print(line)
        file_name = 'policy_eval_results_{}.csv'.format(alg)
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
