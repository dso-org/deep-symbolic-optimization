"""Script used to generate config file."""

import os
import stat
import json
import itertools
from copy import deepcopy
import argparse

def myargparse():
    description = 'Make config json'
    epilog = 'End of documentation'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-bp','--base_path', 
                        type=str,
                        dest='bp',            
                        help="Full path of base config",
                        default='./config/base.json')
    parser.add_argument('-nm','--base_name', 
                        type=str,
                        dest='nm',            
                        help="Name of config file (name.json)",
                        default='base')
    parser.add_argument('-ns','--n_samples', 
                        type=str,
                        dest='ns',            
                        help="n_samples",
                        default='1000000')                        
    parser.add_argument('-bs','--batch_size', 
                        type=str,
                        dest='bs',            
                        help="batch_size",
                        default='500')
    parser.add_argument('-ep','--epsilon', 
                        type=str,
                        dest='ep',            
                        help="epsilon",
                        default='0.1')
    parser.add_argument('-ew','--entropy_weight', 
                        type=str,
                        dest='ew',            
                        help="entropy_weight",
                        default='0.01')                    
    parser.add_argument('-lr','--learning_rate', 
                        type=str,
                        dest='lr',            
                        help="learning_rate",
                        default='1e-3')
    return parser.parse_args()


def create_base(bp,nm,ns,bs,ep,ew,lr):
    
    # benchmarks = [4, 5]
    # mc = 8
    # num_cores = 16    
    
    path = os.path.join(bp)
    with open(path, encoding='utf-8') as f:
        default = json.load(f)

    # Manually turn off saving outputs
    default["training"]["output_file"] = None
    default["training"]["save_all_r"] = False
    # Manually adjust to 1M expressions
    default["training"]["logdir"] = "./log"
    default["training"]["n_samples"] = int(ns)
    default["training"]["batch_size"] = int(bs)
    default["training"]["epsilon"] = float(ep)

    default["controller"]["entropy_weight"] = float(ew)
    default["controller"]["learning_rate"] = float(lr)


    new_config = deepcopy(default)
    path = os.path.join("./", nm + ".json")
    with open(path, 'w') as f:
        json.dump(new_config, f, indent=3)


if __name__ == "__main__":
    args = myargparse()
    create_base(args.bp, args.nm, args.ns, args.bs, args.ep, args.ew, args.lr)
