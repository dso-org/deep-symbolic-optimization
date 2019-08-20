import os
import stat
import json
from copy import deepcopy

ablations = {

    "reward" : {
        "neg_mse" : {
            "training:reward" : "neg_mse"
        },
        "neg_nmse" : {
            "training:reward" : "neg_nmse"
        },
        "neg_nrmse" : {
            "training:reward" : "neg_nrmse"
        },
        "inv_mse" : {
            "training:reward" : "inv_mse"
        },
        "inv_nmse" : {
            "training:reward" : "inv_nmse"
        },
        "inv_nrmse" : {
            "training:reward" : "inv_nrmse"
        },
    },

    "improvements" : {
        "vanilla_no_hierarchical" : {
            # Vanilla
            "controller:entropy_weight" : 0.0,
            "training:b_jumpstart" : False,
            "training:alpha" : 0.0,
            "training:epsilon" : 1.0,
            "training:complexity_weight" : 0.0,
            # No hierarchical            
            "controller:observe_action" : True,
            "controller:observe_parent" : False,
            "controller:observe_sibling" : False
        },
        "vanilla_with_hierarchical" : {
            # Vanilla
            "controller:entropy_weight" : 0.0,
            "training:b_jumpstart" : False,
            "training:alpha" : 0.0,
            "training:epsilon" : 1.0,
            "training:complexity_weight" : 0.0
        },
        "no_entropy" : {
            "controller:entropy_weight" : 0.0
        },
        "no_baseline" : {
            "training:b_jumpstart" : False,
            "training:alpha" : 0.0
        },
        "no_risk" : {
            "training:epsilon" : 1.0
        },
        "with_complexity" : {
            "training:complexity" : "length",
            "training:complexity_weight" : 0.00001
        }
    },

    "constraints" : {
        "no_const" : {
            "controller:constrain_const" : False
        },
        "no_trig" : {
            "controller:constrain_trig" : False
        },
        "no_inv" : {
            "controller:constrain_inv" : False
        },
        "no_min" : {
            "controller:min_length" : 1,
            "controller:constrain_min_len" : False
        },
        "no_max" : {
            "controller:constrain_max_len" : False
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
            "controller:constrain_max_len" : False,
        }
    },

    "other" : {
        "vanilla_no_constraints_no_hierarchical" : {
            # Vanilla
            "controller:entropy_weight" : 0.0,
            "training:b_jumpstart" : False,
            "training:alpha" : 0.0,
            "training:epsilon" : 1.0,
            "training:complexity_weight" : 0.0,
            # No constraints
            "controller:constrain_const" : False,
            "controller:constrain_trig" : False,
            "controller:constrain_inv" : False,
            "controller:min_length" : 1,
            "controller:constrain_min_len" : False,
            "controller:constrain_max_len" : False,
            # No hierarchical
            "controller:observe_action" : True,
            "controller:observe_parent" : False,
            "controller:observe_sibling" : False
        },
        "vanilla_no_constraints_with_hierarchical" : {
            # Vanilla
            "controller:entropy_weight" : 0.0,
            "training:b_jumpstart" : False,
            "training:alpha" : 0.0,
            "training:epsilon" : 1.0,
            "training:complexity_weight" : 0.0,
            # No constraints
            "controller:constrain_const" : False,
            "controller:constrain_trig" : False,
            "controller:constrain_inv" : False,
            "controller:min_length" : 1,
            "controller:constrain_min_len" : False,
            "controller:constrain_max_len" : False
        },
        "full" : {}
    }    
}


def main():
    
    with open("../iclr_config.json", encoding='utf-8') as f:
        template = json.load(f)

    # For each abalation study
    for study, exp in ablations.items():

        # Create the run file for the study
        run_file = os.path.join(study, "run.sh")
        os.makedirs(study, exist_ok=True)
        open(run_file, 'a').close() # Create the file
        st = os.stat(run_file)
        os.chmod(run_file, st.st_mode | stat.S_IEXEC)

        # For each experiment in the study
        for logdir, spec in exp.items():

            config = deepcopy(template)

            path = os.path.join(study, logdir)
            config["training"]["logdir"] = path
            os.makedirs(path, exist_ok=True)

            # Overwrite config parameters
            for k, v in spec.items():
                k = k.split(':')
                assert k[0] in config
                assert k[1] in config[k[0]]
                config[k[0]][k[1]] = v            

            # Save the new config
            with open(os.path.join(path, "config.json"), 'w') as f:
                json.dump(config, f, indent=3)

            # Add the experiment to the study's run file
            with open(run_file, 'a') as f:
                f.write("time python -m dsr.run_benchmarks ./{}/config.json --only=Nguyen --only=Burks --only=Meier --mc=10 --num_cores=16\n".format(path))


if __name__ == "__main__":
    main()