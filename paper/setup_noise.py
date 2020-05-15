"""Script used to generate noise and dataset size parameters from base config file."""

import os
import stat
import json
from copy import deepcopy

import numpy as np


NOISE_SEED_SHIFT = 1000


def main():
    
    with open("config/base.json", encoding='utf-8') as f:
        template = json.load(f)

    # Manually turn off saving all rewards
    template["training"]["save_all_r"] = False
    template["training"]["early_stopping"] = True
    template["gp"]["early_stopping"] = True

    methods = ["gp", "dsr"]
    noise_levels = np.linspace(0.0, 0.1, 11, endpoint=True)
    dataset_size_multipliers = [1, 10]

    # For each method
    for method in methods:

        # Create config directory  
        path = os.path.join("config", "noise", method)
        os.makedirs(path, exist_ok=True)

        # Create the run file
        run_file = "run_noise_{}.sh".format(method)
        open(run_file, 'a').close() # Create the file
        st = os.stat(run_file)
        os.chmod(run_file, st.st_mode | stat.S_IEXEC)

        # For each noise level
        for noise in noise_levels:
            # For each dataset size
            for multiplier in dataset_size_multipliers:
                config = deepcopy(template)
                config["dataset"]["noise"] = noise
                config["dataset"]["dataset_size_multiplier"] = multiplier
                name = "n{}_d{}".format(noise, multiplier)
                logdir = template["training"]["logdir"]
                logdir = os.path.join(logdir, "noise", method, name)
                config["training"]["logdir"] = logdir

                # Save the new config
                with open(os.path.join("config", "noise", method, "{}.json".format(name)), 'w') as f:
                    json.dump(config, f, indent=3)

                # Add the ablation to the run file
                n_cores_task = 24 if method == "dsr" else 32
                with open(run_file, 'a') as f:
                    f.write("time python -m dsr.run ./config/noise/{}/{}.json --method={} --b=Nguyen --mc=10 --seed_shift={} --n_cores_task={}\n".format(method, name, method, NOISE_SEED_SHIFT, num_cores))


if __name__ == "__main__":
    main()

