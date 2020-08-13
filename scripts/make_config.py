"""Script used to generate config file."""

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
    description = 'Make config json for HP study'
    epilog = 'End of documentation'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-bp','--base_path', 
                        type=str,
                        dest='bp',            
                        help="full path of base config",
                        default='../dsr/dsr/config.json')
    parser.add_argument('-nm','--base_name', 
                        type=str,
                        dest='nm',            
                        help="name of config file (name.json)",
                        default='base')
    parser.add_argument('-edd','--extra_data_dir', 
                        type=str,
                        dest='edd',            
                        help="extra_data_dir",
                        default='None') 
    parser.add_argument('-mp','--metric_params', 
                        type=str,
                        dest='mp',            
                        help="metric parameters",
                        default='1.0')
    parser.add_argument('-prtd','--protected', 
                        type=str2bool,
                        dest='prtd',            
                        help="protected",
                        default=True) 
    parser.add_argument('-ns','--n_samples', 
                        type=str,
                        dest='ns',            
                        help="number of samples",
                        default='2000000')                        
    parser.add_argument('-bs','--batch_size', 
                        type=str,
                        dest='bs',            
                        help="batch size",
                        default='500')
    parser.add_argument('-ap','--alpha', 
                        type=str,
                        dest='ap',            
                        help="exponentially-weighted moving average alpha",
                        default='0.5')
    parser.add_argument('-ep','--epsilon', 
                        type=str,
                        dest='ep',            
                        help="risk-seeking fraction epsilon",
                        default='None')
    parser.add_argument('-bl','--baseline', 
                        type=str,
                        dest='bl',            
                        help="baseline",
                        default='R_e')                        
    parser.add_argument('-lr','--learning_rate', 
                        type=str,
                        dest='lr',            
                        help="learning rate",
                        default='1e-3')
    parser.add_argument('-oe','--use_old_entropy', 
                        type=str2bool,
                        dest='oe',            
                        help="use old entropy",
                        default=False)
    parser.add_argument('-ew','--entropy_weight', 
                        type=str,
                        dest='ew',            
                        help="entropy weight",
                        default='0.01')
    parser.add_argument('-pqt','--pqt', 
                        type=str2bool,
                        dest='pqt',            
                        help="use PQT",
                        default=False) 
    parser.add_argument('-pqt_k','--pqt_k', 
                        type=str,
                        dest='pqt_k',            
                        help="length of PQT",
                        default="10")
    parser.add_argument('-pqt_w','--pqt_w',
                        type=str,
                        dest='pqt_w',
                        help="weight of PQT",
                        default="200.0")
    parser.add_argument('-gp_ps','--gp_ps', 
                        type=str,
                        dest='gp_ps',            
                        help="population_size",
                        default="1000")
    parser.add_argument('-gp_ns','--gp_ns', 
                        type=str,
                        dest='gp_ns',            
                        help="gp number of samples",
                        default="2000000")
    parser.add_argument('-gp_ts','--gp_ts', 
                        type=str,
                        dest='gp_ts',            
                        help="gp tournament size",
                        default='2')
    parser.add_argument('-gp_cs','--gp_cs', 
                        type=str,
                        dest='gp_cs',            
                        help="gp crossover probability",
                        default='0.95')
    parser.add_argument('-gp_m','--gp_m', 
                        type=str,
                        dest='gp_m',            
                        help="gp mutation probability",
                        default='0.03')
    parser.add_argument('-gp_prtd','--gp_protected', 
                        type=str2bool,
                        dest='gp_prtd',            
                        help="protected operators",
                        default=True)
    parser.add_argument('-gp_cntrt','--gp_constraint', 
                        type=str2bool,
                        dest='gp_cntrt',            
                        help="constraint mode in gp",
                        default=False)                    
    return parser.parse_args()


def create_base(bp,nm,
                edd,mp,prtd,
                ns,bs,ap,ep,bl,lr,oe,ew,
                pqt,pqt_k,pqt_w,
                gp_ps,gp_ns,gp_ts,gp_cs,gp_m,gp_prtd,gp_cntrt):
         
    path = os.path.join(bp)
    with open(path, encoding='utf-8') as f:
        default = json.load(f)

    # Test names are defined by the command line    
    default["task"]["task_type"] = "regression"
    default["task"]["name"] = None
    default["task"]["dataset"]["file"] = "benchmarks.csv"
    # default["task"]["dataset"]["name"] = None
    # default["task"]["dataset"]["noise"] = None
    if edd == 'None':
        default["task"]["dataset"]["extra_data_dir"] = None
    else:
        default["task"]["dataset"]["extra_data_dir"] = edd
    # default["task"]["dataset"]["function_set"] = None
    # default["task"]["dataset"]["shuffle_data"] = None
    # default["task"]["dataset"]["train_fraction"] = None
    default["task"]["metric"] = "inv_nrmse"
    default["task"]["metric_params"] = [float(mp)]
    default["task"]["threshold"] = 1e-12
    default["task"]["protected"] = prtd

    # Manually adjust to number of expressions
    default["training"]["logdir"] = "./log"
    default["training"]["n_epochs"] = None
    default["training"]["n_samples"] = int(ns)
    default["training"]["batch_size"] = int(bs)
    
    default["training"]["complexity"] = "length"
    default["training"]["complexity_weight"] = 0.0
    
    default["training"]["const_optimizer"] = "scipy"
    default["training"]["const_params"] = {}
    
    default["training"]["alpha"] = float(ap)
    
    if ep == 'None':
        default["training"]["epsilon"] = None
    else:
        default["training"]["epsilon"] = float(ep)
    
    default["training"]["verbose"] = False
    
    default["training"]["baseline"] = bl
    default["training"]["b_jumpstart"] = False
    
    default["training"]["n_cores_batch"] = 1
    default["training"]["summary"] = False
    default["training"]["debug"] = 0
    default["training"]["output_file"] = None
    default["training"]["save_all_r"] = False
    default["training"]["early_stopping"] = True
    default["training"]["hof"] = None

    default["controller"]["cell"] = "lstm"
    default["controller"]["num_layers"] = 1
    default["controller"]["num_units"] = 32
    default["controller"]["initializer"] = "zeros"
    default["controller"]["embedding"] = False
    default["controller"]["embedding_size"] = 8
    default["controller"]["optimizer"] = "adam"
    default["controller"]["learning_rate"] = float(lr)
    
    default["controller"]["observe_action"] = False
    default["controller"]["observe_parent"] = True
    default["controller"]["observe_sibling"] = True
    default["controller"]["constrain_const"] = True
    default["controller"]["constrain_trig"] = True  
    default["controller"]["constrain_inv"] = True  
    default["controller"]["constrain_min_len"] = True  
    default["controller"]["constrain_max_len"] = True      
    default["controller"]["constrain_num_const"] = False 
    
    default["controller"]["use_language_model_prior"] = False
    
    default["controller"]["min_length"] = 4
    default["controller"]["max_length"] = 30  
    default["controller"]["max_const"] = 3  

    default["controller"]["use_old_entropy"] = oe
    default["controller"]["entropy_weight"] = float(ew)
    
    default["controller"]["ppo"] = False
    default["controller"]["ppo_clip_ratio"] = 0.2
    default["controller"]["ppo_n_iters"] = 10
    default["controller"]["ppo_n_mb"] = 4
    
    default["controller"]["pqt"] = pqt
    default["controller"]["pqt_k"] = int(pqt_k)
    default["controller"]["pqt_batch_size"] = 1
    default["controller"]["pqt_weight"] = float(pqt_w)
    default["controller"]["pqt_use_pg"] = False    

    default["gp"]["population_size"] = int(gp_ps)
    default["gp"]["generations"] = None
    default["gp"]["n_samples"] = int(gp_ns)
    default["gp"]["tournament_size"] = int(gp_ts)
    default["gp"]["metric"] = "nmse"
    default["gp"]["const_range"] = [-1.0,1.0]
    default["gp"]["p_crossover"] = float(gp_cs)
    default["gp"]["p_mutate"] = float(gp_m)
    default["gp"]["seed"] = 0
    default["gp"]["early_stopping"] = True
    default["gp"]["threshold"] = 1e-12
    default["gp"]["verbose"] = False
    default["gp"]["protected"] = gp_prtd
    
    if gp_cntrt :
        default["gp"]["constrain_const"] = True
        default["gp"]["constrain_trig"] = True
        default["gp"]["constrain_inv"] = True
        default["gp"]["constrain_min_len"] = True
        default["gp"]["constrain_max_len"] = True
        default["gp"]["constrain_num_const"] = True
    else:
        default["gp"]["constrain_const"] = False
        default["gp"]["constrain_trig"] = False
        default["gp"]["constrain_inv"] = False
        default["gp"]["constrain_min_len"] = False
        default["gp"]["constrain_max_len"] = True
        default["gp"]["constrain_num_const"] = True        
    
    default["gp"]["min_length"] = 4,
    default["gp"]["max_length"] = 30,
    default["gp"]["max_const"] = 3
    
    new_config = deepcopy(default)
    path = os.path.join("./", nm + ".json")
    with open(path, 'w') as f:
        json.dump(new_config, f, indent=3)


if __name__ == "__main__":
    args = myargparse()
    create_base(args.bp, args.nm, 
                args.edd, args.mp, args.prtd,
                args.ns, args.bs, args.ap, args.ep, args.bl, args.lr,
                args.oe, args.ew, 
                args.pqt, args.pqt_k, args.pqt_w,
                args.gp_ps, args.gp_ns, args.gp_ts, args.gp_cs, args.gp_m, args.gp_prtd, args.gp_cntrt)
