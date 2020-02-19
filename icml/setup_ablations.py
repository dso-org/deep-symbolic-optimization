"""Script used to generate ablations parameters from base config file."""

import os
import stat
import json
from copy import deepcopy

ABLATIONS_SEED_SHIFT = 100

ablations = {
    "vanilla" : {
        "controller:entropy_weight" : 0.0,
        "training:baseline" : "ewma_R",
        "training:b_jumpstart" : False,
        "training:alpha" : 0.5,
        "training:epsilon" : 1.0,
        "training:complexity_weight" : 0.0,
        "controller:observe_action" : True,
        "controller:observe_parent" : False,
        "controller:observe_sibling" : False,
        "controller:constrain_const" : False,
        "controller:constrain_trig" : False,
        "controller:constrain_inv" : False,
        "controller:min_length" : 1,
        "controller:constrain_min_len" : False,
        "controller:constrain_max_len" : False
    },
    "no_improvements" : {
        "controller:entropy_weight" : 0.0,
        "training:baseline" : "ewma_R",
        "training:b_jumpstart" : False,
        "training:alpha" : 0.5,
        "training:epsilon" : 1.0,
        "training:complexity_weight" : 0.0,
        "controller:observe_action" : True,
        "controller:observe_parent" : False,
        "controller:observe_sibling" : False
    },
    "no_hierarchical" : {
        "controller:observe_action" : True,
        "controller:observe_parent" : False,
        "controller:observe_sibling" : False
    },
    "no_entropy" : {
        "controller:entropy_weight" : 0.0
    },
    "no_risk" : {
        "training:epsilon" : 1.0,
        "training:baseline" : "ewma_R",
        "training:alpha" : 0.5,
        "training:b_jumpstart" : False,
    },
    "no_trig" : {
        "controller:constrain_trig" : False
    },
    "no_inv" : {
        "controller:constrain_inv" : False
    },
    "no_min_max" : {
        "controller:min_length" : 1,
        "controller:constrain_min_len" : False,
        "controller:constrain_max_len" : False
    },
    "no_constraints" : {
        "controller:constrain_const" : False,
        "controller:constrain_trig" : False,
        "controller:constrain_inv" : False,
        "controller:min_length" : 1,
        "controller:constrain_min_len" : False,
        "controller:constrain_max_len" : False
    },
    "full" : {}, # No ablations; DSR
}


def main():
    
    with open("config/base.json", encoding='utf-8') as f:
        template = json.load(f)

    # Create config directory
    path = os.path.join("config", "ablations")
    os.makedirs(path, exist_ok=True)

    # Manually turn off saving all rewards
    template["training"]["save_all_r"] = False
    template["training"]["early_stopping"] = True
    template["deap"]["early_stopping"] = True

    # Create the run file
    run_file = "run_ablations.sh"
    open(run_file, 'a').close() # Create the file
    st = os.stat(run_file)
    os.chmod(run_file, st.st_mode | stat.S_IEXEC)

    # For each abalation
    for name, spec in ablations.items():

        config = deepcopy(template)

        logdir = template["training"]["logdir"]
        logdir = os.path.join(logdir, "ablations", name)
        config["training"]["logdir"] = logdir

        # Overwrite config parameters
        for k, v in spec.items():
            k = k.split(':')
            assert k[0] in config
            assert k[1] in config[k[0]], (k[1], config[k[0]])
            config[k[0]][k[1]] = v            

        # Save the new config
        with open(os.path.join("config", "ablations", "{}.json".format(name)), 'w') as f:
            json.dump(config, f, indent=3)

        # Add the ablation to the run file
        with open(run_file, 'a') as f:
            f.write("time python -m dsr.run ./config/ablations/{}.json --method=dsr --b=Nguyen --mc=10 --seed_shift={} --num_cores=24\n".format(name, ABLATIONS_SEED_SHIFT))


if __name__ == "__main__":
    main()
