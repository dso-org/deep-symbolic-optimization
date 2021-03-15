import numpy as np
from dsr.const import make_const_optimizer
from dsr.program import Program
from dsr.gp import tokens as gp_tokens

try:
    from deap import gp
    from deap import base
    from deap import tools
    from deap import creator
    from deap import algorithms
except ImportError:
    gp          = None
    base        = None
    tools       = None
    creator     = None
    algorithms  = None
    
try:
    CONST_TOKEN = Program.library.names.index("const")
except:
    CONST_TOKEN = None

def set_const_individuals(const_idxs, consts, individual):
        
    # These are optimizable constants, not user constants. 
    for i, const in zip(const_idxs, consts):
        individual[i] = gp_tokens.Terminal(const, False, object, token=CONST_TOKEN)
        individual[i].name = "mutable_const_{}".format(i) 
        
    return individual

    
def reset_consts(pset_mapping, val=1.0):
        
    for k, v in pset_mapping.items():
        if v.name.startswith("mutable_const_"):
            v.value = val # 1.0
            
    return pset_mapping


def get_consts():
    
    user_consts         = [t for i, t in enumerate(Program.library.tokens) if t.arity == 0 and t.input_var is None and t.name != "const"] 
    mutable_consts      = len([t for i, t in enumerate(Program.library.tokens) if t.name == "const"])
    
    return user_consts, mutable_consts
        

def const_opt(pset, mutable_consts, max_const, user_consts, const_params, config_training):
    
    # Are we optimizing a const?               
    if mutable_consts:
        print("-------------------------------------")
        print("DEAP installing constants to be optimized:")
        const_optimizer  = config_training["const_optimizer"] # Probably SciPy
        
        # Need to differentiate between mutable and non mutable const
        const_params    = const_params if const_params is not None else {}
        const_opt       = make_const_optimizer(const_optimizer, **const_params)
        for i in range(max_const):
            dname   = "mutable_const_{}".format(i)
            dvalue  = np.float(1.0)
            pset.addTerminal(dvalue, name=dname)
            pset.mapping[dname].value = dvalue
            print("\t{} is {}".format(dname, pset.mapping[dname].value))
        print("-------------------------------------")
    else:
        const_opt       = None   
    
    # Add user provided constants.
    if len(user_consts) > 0:
        print("-------------------------------------")
        print("DEAP installing user supplied constants:")
        for i,v in enumerate(user_consts):
            # Note, t.function() will return value
            dname   = "user_const_{}".format(v.name)
            dvalue  = np.float(v.function())            
            pset.addTerminal(dvalue, name=dname)
            pset.mapping[dname].value = dvalue 
            print("\t{}".format(pset.mapping[dname].value))
        print("-------------------------------------")
    
    return pset, const_opt






