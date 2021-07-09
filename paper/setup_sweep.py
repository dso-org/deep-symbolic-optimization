"""Script used to generate sweep parameters from base config file."""

import os
import stat
import json
import itertools
from copy import deepcopy


def main():
    
    path = os.path.join("config", "base.json")
    with open(path, encoding='utf-8') as f:
        default = json.load(f)

    # Manually turn off saving outputs
    default["training"]["output_file"] = None
    default["training"]["save_all_r"] = False

    # Manually adjust to 1M expressions
    default["training"]["n_samples"] = 1000000
    default["gp"]["n_samples"] = 1000000

    benchmarks = [4, 5]
    mc = 8
    n_cores_task = 16

    sweep_dso = {
        "training" : {
            "batch_size" : [250, 500, 1000],
            "epsilon" : [0.05, 0.1, 0.15]
        },
        "controller" : {
            "entropy_weight" : [0.01, 0.05, 0.1],
            "learning_rate" : [3e-4, 5e-4, 1e-3]
        }
    }

    sweep_gp = {
        "gp" : {
            "population_size" : [100, 250, 500, 1000],
            "tournament_size" : [2, 3, 5, 10],
            "p_crossover" : [0.25, 0.5, 0.75, 0.9, 0.95],
            "p_mutate" : [0.01, 0.03, 0.05, 0.10, 0.15]
        }
    }

    methods = ["dso", "gp"]
    sweeps = [sweep_dso, sweep_gp]

    for method, sweep in zip(methods, sweeps):

        run_file = "run_sweep_{}.sh".format(method)

        flat = {}
        result = {}
        for k, v in sweep.items():
            result[k] = {}
            flat.update(v)

        all_keys = []
        all_vals = []

        for key, val in flat.items():
            all_keys.append(key)
            all_vals.append(val)

        all_combos = list(itertools.product(*all_vals))
        flat_configs = [{all_keys[i] : combo[i] for i in range(len(all_keys))} for combo in all_combos]

        new_configs = []
        for i, flat_config in enumerate(flat_configs):
            name = "sweep_{}".format(i)
            new_config = deepcopy(default)
            for key in sweep.keys():
                new_config[key].update({k:v for k,v in flat_config.items() if k in sweep[key]})
            new_config["training"]["logdir"] = os.path.join("sweep", method, name)
            new_configs.append(new_config)
            path = os.path.join("config", "sweep", method)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "{}.json".format(name))
            with open(path, 'w') as f:
                json.dump(new_config, f, indent=3)
        
            with open(run_file, 'a') as f:
                only = " ".join(["--b=Nguyen-{}".format(b) for b in benchmarks])
                cmd = "time python -m dso.run {} --method={} {} --mc={} --n_cores_task={}\n".format(path, method, only, mc, num_cores)
                f.write(cmd)

        # Make the run file executable
        st = os.stat(run_file)
        os.chmod(run_file, st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    main()
