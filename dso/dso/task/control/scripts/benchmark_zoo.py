"""Evaluate a Zoo policy on a Gym environment."""

import os
import sys

import click
import pandas as pd
import numpy as np

from dso.program import Program, from_str_tokens
from dso.task import set_task
import dso.task.control # Register custom envs
import dso.task.control.utils as U

DEFAULT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
POSITIVE_SCALES = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

ENVIRONMENTS = {
    "CustomContinuousCartPole-v0": {
        "n_actions" : 1,
        "env_kwargs" : {
            "dt_multiplier" : DEFAULT_SCALES
        }
    }, 
    "MountainCarContinuous-v0" : {
        "n_actions" : 1,
        "env_kwargs" : {
            "power_multiplier" : DEFAULT_SCALES,
            "starting_state_multiplier" : POSITIVE_SCALES
        },
        # "symbolic" : ["div,mul,0.05,log,x2,add,x2,log,mul,10.0,x2"]
        "symbolic" : ["div,log,cos,1.0,log,x2"] # 99.09
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
        "symbolic" : ["add,mul,-2.0,x2,div,add,mul,-8.0,x2,mul,-2.0,x3,x1"]
    },
    "InvertedDoublePendulumBulletEnv-v0" : {
        "n_actions" : 1,
        "env_kwargs" : {}
    },
    "InvertedPendulumSwingupBulletEnv-v0" : {
        "n_actions" : 1,
        "env_kwargs" : {}
    },
    "LunarLanderContinuous-v2" : {
        "n_actions" : 2,
        "env_kwargs" : {}
    },
    "ReacherBulletEnv-v0" : {
        "n_actions" : 2,
        "env_kwargs" : {}
    },
    "HopperBulletEnv-v0" : {
        "n_actions" : 3,
        "env_kwargs" : {}
    }
}

ALGORITHMS = [
    'a2c',
    'acktr',
    'ddpg',
    'sac',
    'ppo2',
    'trpo',
    'td3',
    'symbolic'
]


@click.command()
@click.argument("zoo_root", type=str)
@click.option("--alg", multiple=True, type=str, help="Which algorithms to run (default: all).")
@click.option("--env", multiple=True, type=str, help="Which environments to run (default: all).")
@click.option("--n_episodes", type=int, default=100, help="Number of evaluation episodes.")
@click.option("--output_filename", type=str, default="benchmark_zoo_results.csv", help="Filename to output results.")
@click.option("--robustness", is_flag=True, default=False, help="Sweep across env kwargs.")
def main(zoo_root, alg, env, n_episodes, output_filename, robustness):

    # Rename arguments
    ZOO_ROOT = zoo_root
    algorithms = alg
    environments = env

    # Add ZOO_ROOT to Python path (required to load SAC Bullet models)
    sys.path.append(ZOO_ROOT)

    # Default: run all algorithms/environments
    if len(algorithms) == 0:
        algorithms = ALGORITHMS
    if len(environments) == 0:
        environments = list(ENVIRONMENTS.keys())

    # Write header
    if not os.path.isfile(output_filename):
        pd.DataFrame({"Environment" : [],
                    "Algorithm" : [],
                    "Score" : [],
                    "Episodes" : [],
                    "Parameter" : [],
                    "Value" : []}).to_csv(output_filename, index=False)

    for alg in algorithms:

        if alg != "symbolic":
            # Pre-trained policies
            path = os.path.join(ZOO_ROOT, "trained_agents", alg)
            contents = os.listdir(path)
            files = [f for f in contents if os.path.isfile(os.path.join(path, f))]

        for env in environments:

            if alg != "symbolic":

                # Look for pre-trained policies
                found = False
                for f in files:
                    if f.startswith(env):
                        anchor = os.path.join(path, f)
                        found = True
                        break
                if not found:
                    print("Could not find pre-trained policy for environment {} for algorithm {}.".format(env, alg))

                    # Try looking again for manually trained policies
                    path = os.path.join(ZOO_ROOT, "logs", alg, env + "_1")
                    contents = os.listdir(path)
                    files = [f for f in contents if os.path.isfile(os.path.join(path, f))]
                    for f in files:
                        if f == "best_model.zip":
                            found = True
                            anchor = os.path.join(path, f)
                            break

                    if not found:
                        print("Could not find manually trained policy for {} for algorithm {}.".format(env, alg))
                        continue

            n_actions = ENVIRONMENTS[env]["n_actions"]

            if alg == "symbolic":
                anchor = None
                action_spec = ENVIRONMENTS[env]["symbolic"]
            else:
                action_spec = ["anchor"] * n_actions

            # Generate env_kwarg combinations.
            env_kwargs_combinations = [{}]
            name = env
            if robustness:
                if not name.startswith("Custom"):
                    name = "Custom" + name
                for key, values in ENVIRONMENTS[env]["env_kwargs"].items():
                    for val in values:
                        env_kwargs_combinations.append({key : val})

            for env_kwargs in env_kwargs_combinations:
                config_task = {
                    "task_type" : "control",
                    "env" : name,
                    "anchor" : anchor,
                    "algorithm" : alg if anchor is not None else None,
                    "action_spec" : action_spec,
                    "n_episodes_test" : n_episodes,
                    "success_score" : 200.0,
                    "function_set" : ["add","sub","mul","div","sin","cos","exp","log","const"],
                    #"zoo_root" : ZOO_ROOT,
                    "env_kwargs" : env_kwargs,
                    "protected" : False
                }

                # Generate the eval_function
                set_task(config_task)
                Program.clear_cache()

                # Create dummy Program, whose eval function does not use p.execute()
                p = Program(np.array([]))

                # Evaluate the Zoo policy
                score = p.evaluate["r_avg_test"]

                # Extract parameter and value
                if len(env_kwargs) > 0:
                    parameter = list(env_kwargs.keys())[0]
                    value = list(env_kwargs.values())[0]
                else:
                    parameter = "N/A"
                    value = "N/A"

                # Write result
                print("Finished evaluating {} on {}. Score: {}.".format(alg.upper(), name, score)) 
                df = pd.DataFrame({"Environment" : [name],
                                   "Algorithm" : [alg],
                                   "Score" : [score],
                                   "Episodes" : [n_episodes],
                                   "Parameter" : [parameter],
                                   "Value" : [value]})
                df.to_csv(output_filename, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
