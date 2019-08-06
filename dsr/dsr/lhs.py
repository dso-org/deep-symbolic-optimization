import os
import stat
import json
import itertools
from copy import deepcopy
import sys

import click
import numpy as np
from scipy.stats import multinomial
from pyDOE import lhs

np.random.seed(0)

abbrev_key = {
    "batch_size" : "bs",
    "complexity_weight" : "cw",
    "num_units" : "lstm",
    "max_length" : "l",
    "alpha" : "a",
    "epsilon" : "e",
    "entropy_weight" : "ent",
    "learning_rate" : "lr",
    "ppo_n_iters" : "it",
    "ppo_n_mb" : "mb",
    "embedding" : "emb",
    "embedding_size" : "emb",
    "b_jumpstart" : "b",
    "reward" : "r",
    "constrain_trig" : "ctrig",
    "constrain_const" : "cconst",
    "constrain_inv" : "cinv",
    "observe_action" : "ac"
}

abbrev_val = {
    True : "T",
    False : "F",
    "neg_mse" : "Neg",
    "inv_mse" : "Inv"
}

def abbrev(k, v):
    a = ""
    if k in abbrev_key:
        a += abbrev_key[k]
    else:
        a += k.strip('_')
    if v in abbrev_val:
        a += abbrev_val[v]
    elif isinstance(v, str):
        a += v
    else:
        a += "{:g}".format(v)
    return a

def generate_configs_lhs(default, sweep, n):
    """
    Generates config files using Latin hypercube sampling.

    Given default and sweep JSON config files, create a Latin hypercube of
    config files, create experiment directories with config files and a run
    script.

    Parameters
    __________

    default : dict
        Default hyperparameters.

    sweep : str
        Specifications of sweep hyperparameters. Supports dict with 'low' and
        'high', dict with 'start', 'stop', and 'n' (or 'step'), and list of
        discrete values.

    n : int
        Number of Latin hypercube samples.
    """

    expdir = default["training"]["logdir"]
    run_file = os.path.join("log", expdir, "run.sh")

    # Build a dictionary from (top-key, bottom-key) to list of values
    params = {}
    blank = deepcopy(default)
    for k1,v1 in sweep.items():
        for k2,v2 in sweep[k1].items():
            params[(k1, k2)] = v2
            blank[k1][k2] = None

    # Generate LHS config dictionaries
    configs = [deepcopy(blank) for _ in range(n)]
    lh = lhs(len(params), samples=n, criterion="center")
    logdirs = {}
    for col, (k, v) in enumerate(params.items()):

        # Continuous value between low and high
        if isinstance(v, dict) and "low" in v.keys():
            v = np.linspace(start=v["low"],
                            stop=v["high"],
                            num=n+1,
                            endpoint=True)

        # Evenly spaced interval
        elif isinstance(v, dict) and "start" in v and "stop" in v:
            if "n" in v:
                num = v["n"]
            elif "step" in v:
                num = round((v["stop"] - v["start"])/v["step"]) + 1
            v = np.linspace(start=v["start"],
                            stop=v["stop"],
                            num=num,
                            endpoint=True)

        # Discrete set of hard-coded values
        elif isinstance(v, list):
            v = np.array(v)

        else:
            raise ValueError("Unrecognized parameter specification.")

        v = np.array(v)
        n = len(v)
        indices = (n*lh[:, col]).astype(int)
        points = v[indices].tolist()

        for i, p in enumerate(points):
            configs[i][k[0]][k[1]] = p

    # Save configs to file
    for config in configs:
        logdir = '_'.join([abbrev(k2,v2) for k1 in config.keys() for k2,v2 in config[k1].items() if (k1,k2) in params])
        if logdir in logdirs:
            logdirs[logdir] += 1
        else:
            logdirs[logdir] = 0
        logdir += '_' + str(logdirs[logdir]) # Counter for possibly repeated hyperparameter combinations
        config["training"]["logdir"] = os.path.join(expdir, logdir)
        path = os.path.join("log", expdir, logdir)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f, indent=3)
        with open(run_file, 'a') as f:
            f.write("time python run_benchmarks.py ./{}/config.json\n".format(path))
    
    # Make the run file executable
    st = os.stat(run_file)
    os.chmod(run_file, st.st_mode | stat.S_IEXEC)


# def generate_configs():
    
#     config_filename = "koza_config.json"
#     with open(config_filename, encoding='utf-8') as f:
#         default = json.load(f)

#     sweep = {
#         "training" : {
#             "alpha" : [0.1, 0.01],
#             "epsilon" : [0.1, 0.25]
#         },
#         "controller" : {
#             "entropy_weight" : [0.0, 0.01],
#             "learning_rate" : [1e-3, 1e-4],
#             "ppo" : [True, False],
#             "embedding" : [True, False]
#         }
#     }

#     abbrev_key = {
#         "batch_size" : "bs",
#         "complexity_weight" : "comp",
#         "num_units" : "lstm",
#         "max_length" : "l",
#         "alpha" : "a",
#         "epsilon" : "e",
#         "entropy_weight" : "ent",
#         "learning_rate" : "lr",
#         "embedding" : "emb",
#     }

#     abbrev_val = {
#         True : "T",
#         False : "F"
#     }

#     def abbrev(k, v):
#         a = ""
#         if k in abbrev_key:
#             a += abbrev_key[k]
#         else:
#             a += k.strip('_')
#         if v in abbrev_val and isinstance(v, bool):
#             a += abbrev_val[v]
#         else:
#             a += "{:.5f}".format(v)
#         return a
#     # abbrev = lambda k,v : (abbrev_key[k] if k in abbrev_key else k.strip('_')) + (abbrev_val[v] if v in abbrev_val and isinstance(v, bool) else str(v))

#     expdir = default["training"]["logdir"]

#     flat = {}
#     result = {}
#     for k, v in sweep.items():
#         result[k] = {}
#         flat.update(v)

#     all_keys = []
#     all_vals = []

#     for key, val in flat.items():
#         if isinstance(val, dict):
#             all_keys.append(key)
#             if "start" not in val or "stop" not in val:
#                 raise ValueError("Dictionary entry must include values for 'start' and 'stop'.")
#             if "step" in val and "num" in val:
#                 raise ValueError("Dictionary entry must include values for either 'step' and 'num', but not both.")
#             if "step" in val:
#                 all_vals.append(np.arange(**val))
#             elif "num" in val:
#                 all_vals.append(np.linspace(**val))
#             else:
#                 raise ValueError("Dictionary entry must include values for either 'step' or 'num'.")
#         elif isinstance(val, list):
#             all_keys.append(key)
#             all_vals.append(val)
#         else: # Scalar
#             all_keys.append(key)
#             all_vals.append(val)

#     all_combos = list(itertools.product(*all_vals))
#     flat_configs = [{all_keys[i] : combo[i] for i in range(len(all_keys))} for combo in all_combos]

#     new_configs = []
#     for flat_config in flat_configs:
#         new_config = deepcopy(default)
#         for key in sweep.keys():
#             new_config[key].update({k:v for k,v in flat_config.items() if k in sweep[key]})
#         logdir = '_'.join([abbrev(k,v) for k,v in flat_config.items()])
#         new_config["training"]["logdir"] = os.path.join(expdir, logdir)
#         new_configs.append(new_config)
#         path = os.path.join("log", expdir, logdir)
#         os.makedirs(path, exist_ok=True)
#         with open(os.path.join(path, "config.json"), 'w') as f:
#             json.dump(new_config, f, indent=3)

#     return new_configs


@click.command()
@click.option('--default', default="config.json", help="JSON filename of default hyperparameters")
@click.option('--sweep', default="sweep.json", help="JSON filename of sweep hyperparameter specifications")
@click.option('--n', default=100, type=int, help="Number of Latin hypercube samples")
def main(default, sweep, n):

    with open(default, encoding='utf-8') as f:
        default = json.load(f)

    with open(sweep, encoding='utf-8') as f:
        sweep = json.load(f)

    generate_configs_lhs(default, sweep, n)


if __name__ == "__main__":
    main()