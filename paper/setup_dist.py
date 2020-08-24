"""Script used to generate risk-seeking vs standard policy gradient experiments from base config file."""

import os
import stat
import json
from copy import deepcopy


experiments = {
    "risk-seeking" : {},
    "standard" : {
        "training:epsilon" : 1.0,
        "training:baseline" : "ewma_R",
        "training:alpha" : 0.5,
        "training:b_jumpstart" : False,
    }
}


def main():
    
    with open("config/base.json", encoding='utf-8') as f:
        template = json.load(f)

    # Create config directory
    path = os.path.join("config", "dist")
    os.makedirs(path, exist_ok=True)

    # Turn on saving all rewards
    template["training"]["save_all_r"] = True

    # Turn off early stopping
    template["training"]["early_stopping"] = False

    # Create the run file
    run_file = "run_dist.sh"
    open(run_file, 'a').close() # Create the file
    st = os.stat(run_file)
    os.chmod(run_file, st.st_mode | stat.S_IEXEC)

    # For each experiment
    for name, spec in experiments.items():

        config = deepcopy(template)

        logdir = template["training"]["logdir"]
        logdir = os.path.join(logdir, "dist", name)
        config["training"]["logdir"] = logdir

        # Overwrite config parameters
        for k, v in spec.items():
            k = k.split(':')
            assert k[0] in config
            assert k[1] in config[k[0]], (k[1], config[k[0]])
            config[k[0]][k[1]] = v            

        # Save the new config
        with open(os.path.join("config", "dist", "{}.json".format(name)), 'w') as f:
            json.dump(config, f, indent=3)

        # Add the experiment to the run file
        with open(run_file, 'a') as f:
            f.write("time python -m dsr.run ./config/dist/{}.json --method=dsr --b=Nguyen --mc=100 --n_cores_task=24\n".format(name))


if __name__ == "__main__":
    main()
