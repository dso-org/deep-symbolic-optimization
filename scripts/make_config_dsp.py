"""
Script used to generate config file.

User case:
    If you have an old input deck with a config file that is stale (it does not 
    contain fields that have been recently added), you can do:
    
        python3 $(hypothesis_testing)/scripts/make_config_dsp.py -bp 
        $(hypothesis_testing)/dso/dso/config_dsp.json -out my_own_config.json -bs 100
        
    to get a fresh version of config.json with your parameters of choice
"""

import os
import stat
import json
import itertools
from copy import deepcopy
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
    parser.add_argument('-bp','--base_path', 
                        type=str,
                        dest='bp',            
                        help="full path of base config",
                        default='../dso/dso/config.json')
    parser.add_argument('-out','--output_name', 
                        type=str,
                        dest='out',            
                        help="name of output config file (name.json)",
                        default='out_base')
    parser.add_argument('-fs','--fix_seeds', 
                        type=str2bool,
                        dest='fs',            
                        help="name of output config file (name.json)",
                        default=True)
    parser.add_argument('-et','--n_episodes_train', 
                        type=int,
                        dest='et',            
                        help="n_episodes_train",
                        default=10)
    parser.add_argument('-es','--episode_seed_shift', 
                        type=int,
                        dest='es',            
                        help="episode_seed_shift",
                        default=0)  
    parser.add_argument('-bs','--batch_size', 
                        type=int,
                        dest='bs',            
                        help="batch_size",
                        default=100)
    parser.add_argument('-ep','--epsilon', 
                        type=float,
                        dest='ep',            
                        help="epsilon",
                        default=0.1)
    parser.add_argument('-lr','--learning_rate', 
                        type=float,
                        dest='lr',            
                        help="learning_rate",
                        default=0.001)
    parser.add_argument('-ew','--entropy_weight', 
                        type=float,
                        dest='ew',            
                        help="entropy_weight",
                        default=0.001) 
    return parser.parse_args()

params = {
    "fs"  : ("task", "fix_seeds", bool),
    "et"  : ("task", "n_episodes_train", int),
    "es"  : ("task", "episode_seed_shift", int),
    "bs"  : ("training", "batch_size", int),
    "ep"  : ("training", "epsilon", float),
    "lr"  : ("controller", "learning_rate", float),
    "ew"  : ("controller", "entropy_weight", float)
}

def make_config(**kwargs):

    path = os.path.join(kwargs['bp'])
    with open(path, encoding='utf-8') as f:
        default = json.load(f)
    
    new_config = deepcopy(default)
    
    for hp, spec in params.items():
        value = kwargs[hp]
        if isinstance(spec, tuple):
            new_config[spec[0]][spec[1]] = spec[2](value)
        else:
            for subspec in spec[value]:
                new_config[subspec[0]][subspec[1]] = subspec[2]
     
    path = os.path.join("./", kwargs['out'] + ".json")
    with open(path, 'w') as f:
        json.dump(new_config, f, indent=3)


if __name__ == "__main__":
    args = myargparse()
    make_config(**args.__dict__)
