"""
Given a starting config with a completed HOF, generate the config
for the next action.

User case:

    When manually going to train the next action is dsp, you can do:
    
        python3 $(hypothesis_testing)/scripts/next_action_config_from_log.py 
        -lp log_of_previous_action -out next_action_dir
        
    to get the config file for the next action (with the best expression 
    inserted in the file to use as anchor)
"""

import os
import sys
import json
from copy import deepcopy
import pandas as pd
import argparse

# Function for Boolean type in the arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def myargparse():
    description = 'Make config json for HP study in DSP'
    epilog = 'End of documentation'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-lp','--log_path', 
                        type=str,
                        dest='lp',            
                        help="full path of previous action log directory (e.x., log_2020-09-14-153920)",
                        default='log_2020-09-14-153920')
    parser.add_argument('-op','--out_path', 
                        type=str,
                        dest='op',            
                        help="name of directory with next config json",
                        default='next_action_dir')        
    return parser.parse_args()


def make_config(**kwargs):
    # Load the previous config
    # This should be the path to the verbose config that includes the datestamp in logdir
    exp_dir = kwargs['lp']
    prev_config_path = os.path.join(exp_dir, "config.json")
    with open(prev_config_path, 'r') as f:
        prev_config = json.load(f)
    # Extract seed (assuming one seed per job)
    cmd_path = os.path.join(exp_dir, "cmd.out")
    with open(cmd_path, 'r') as f:
        cmd = f.readlines()[0]
    if "seed_shift" in cmd:
        seed = int(cmd.split("seed_shift")[1][1:].strip().split()[0])
    else:
        seed = 0
    
    # Extract env and logdir
    env = prev_config["task"]["name"]
    prev_logdir = prev_config["training"]["logdir"]
    assert "_202" in prev_logdir, "Config should include datestamp in logdir."
    # Extract action ID and number of actions
    prev_action_spec = prev_config["task"]["action_spec"]
    n_actions = len(prev_action_spec)
    prev_action = prev_action_spec.index(None)
    action = prev_action + 1
    # Calling the script to compile the entire symbolic multi-action policy
    if action == n_actions : 
        print("Warning: Already ran DSP for all actions. The output config \
file will contain the entire symbolic multi-action policy. Note that \
the output config file cannot be used to run DSP.")
    assert action <= n_actions, "You are running the script over a \
filled symbolic multi-action policy config file."
    
    # Read the HOF
    files = os.listdir(prev_logdir)
    hof_filename = "dsr_{}_{}_hof.csv".format(env, seed)
    assert hof_filename in files, "HOF not found; run did not complete."
    df = pd.read_csv(os.path.join(prev_logdir, hof_filename))
    # Extract the best expression for prev_action
    best_score = df["r_avg_test"].max()
    best_traversal = df.loc[df["r_avg_test"] == best_score]["traversal"].values[0]
    # Update the new config action_spec
    config = deepcopy(prev_config)
    config["task"]["action_spec"][prev_action] = best_traversal.split(',')
    if action < n_actions : 
        config["task"]["action_spec"][action] = None
    # Update the new config logdir
    logdir = prev_logdir.split("_202")[0]
    before = "a{}".format(prev_action)
    after = "a{}".format(action)
    logdir = logdir.replace(before, after)
    
    # Save config
    config_filename = "config_{}.json".format(os.path.basename(logdir))
    path = os.path.join(kwargs['op'], config_filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=3)

if __name__ == "__main__":
    args = myargparse()
    make_config(**args.__dict__)
