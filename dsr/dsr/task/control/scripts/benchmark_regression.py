"""Evaluates a purely symbolic policy on the environment."""

import os
from copy import deepcopy

import click
import pandas as pd

from dsr.program import Program
from dsr.task import make_task


ENVIRONMENTS = {
    "MountainCarContinuous-v0" : 1,
    "Pendulum-v0" : 1,
    "InvertedDoublePendulumBulletEnv-v0" : 1,
    "InvertedPendulumSwingupBulletEnv-v0" : 1,
    "LunarLanderContinuous-v2" : 2,
    "ReacherBulletEnv-v0" : 2,
    "HopperBulletEnv-v0" : 3
}
MAX_ACTIONS = max(ENVIRONMENTS.values())


def eval_task(config_task):

    # Create the task
    protected = config_task["protected"]
    Program.set_execute(protected)
    reward_function, eval_function, function_set, n_input_var, stochastic, extra_info = make_task(**config_task)
    Program.set_reward_function(reward_function)
    Program.set_eval_function(eval_function)

    p = list(extra_info["symbolic_actions"].values())[0]

    return p.evaluate


@click.command()
@click.argument("exp_dir", type=str, default=".") # Directory containing HOF results
@click.option("--env", multiple=True, type=str, default=("all",), help="Name of environment")
@click.option("--mc", type=int, default=10, help="Number of seeds.")
@click.option("--output_filename", type=str, default="benchmark_regression.csv", help="Name of output file.")
@click.option("--n_episodes", type=int, default=1000, help="Number of episodes to evaluate per env.")
def main(exp_dir, env, mc, output_filename, n_episodes):

    environments = env # Rename variable

    if environments[0] == "all":
        environments = list(ENVIRONMENTS.keys())

    header = ["env", "seed"]
    header += ["traversal_{}".format(i) for i in range(MAX_ACTIONS)]
    header += ["expression_{}".format(i) for i in range(MAX_ACTIONS)]
    header += ["n_episodes", "score"]

    # Write header
    if not os.path.isfile(output_filename):
        data = {s : [] for s in header}
        pd.DataFrame(data).to_csv(output_filename, index=False)

    config_template = {
        "task_type" : "control",
        "env_kwargs" : {},
        "n_episodes_train" : 2,
        "n_episodes_test" : n_episodes,
        "success_score": 200.0, # Dummy; unused
        "stochastic" : False,
        "protected" : False,
        "function_set" : ["add", "mul", "sub", "div", "sin", "cos", "exp", "log", "const"]
    }

    for seed in range(mc):
        for env in environments:

            # Read the data and extract the best expressions
            traversals = []
            expressions = []
            n_actions = ENVIRONMENTS[env]
            for action in range(n_actions):
                f = "dsr_{}_a{}_{}_hof.csv".format(env, action, seed)
                path = os.path.join(exp_dir, f)
                df = pd.read_csv(path)

                # Extract expression with the best evaluation
                best_row = df.loc[df["nmse_test"].idxmax()]

                traversals.append(best_row["traversal"])
                expressions.append(best_row["expression"])

            # Run the environment with the best expressions
            config_task = deepcopy(config_template)
            config_task["name"] = env
            config_task["action_spec"] = traversals
            score = eval_task(config_task)["r_avg_test"]

            # Add missing values
            for action in range(n_actions, MAX_ACTIONS):
                traversals.append(None)
                expressions.append(None)

            data = [env, seed] + traversals + expressions + [n_episodes, score]
            data = dict(zip(header, data))
            pd.DataFrame([data]).to_csv(output_filename, mode='a', header=False, index=False)


if __name__ == "__main__":
    main()