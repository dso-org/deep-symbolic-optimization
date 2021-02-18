import numpy as np
from dsr.program import Program,  _finish_tokens

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
    
r"""
    This is a base class for accessing DEAP and interfacing it with DSR. 
        
    These are pure symblic components which relate to any symblic task. These are not purely task agnostic and
    are kept seprate from core.
"""

def DEAP_to_math_tokens(individual, tokens_size):
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(individual, gp.PrimitiveTree), "Program tokens should be a Deap GP PrimativeTree object."

    l = min(len(individual),tokens_size)
  
    tokens              = np.zeros(tokens_size,dtype=np.int32)
    optimized_consts    = []
    
    for i in range(l):
        
        t = individual[i]
        
        if isinstance(t, gp.Terminal):
            if t.name.startswith("user_const_"):
                '''
                    User provided constants which do not change.
                '''
                # The ID back to DSR terminal token is stored in the name
                tokens[i]   = Program.library.names.index(t.name.split('_')[2])
            elif t.name.startswith("mutable_const_"):
                '''
                    Optimizable contstants which we can call the optimizer on.
                '''
                # Get the constant token, this will not store the actual const. It however is in the lib tokens. 
                tokens[i]                   = Program.library.names.index("const")
                optimized_consts.append(t.value)
            else:
                '''
                    Arg tokens also known as X.
                '''
                # Get the int which is contained in "ARG{}",
                # Here is it is not x{} since the rename function does not change the internal node name. 
                # This is due to the name being recast through str() when the terminal sets its own name. 
                # Most likely this is a bug in DEAP
                input_var   = int(t.name[3:])
                tokens[i]   = input_var         
        else:
            '''
                Function tokens such as sin, log and multiply 
            '''
            # Get the index number for this op from the op list in Program.library
            tokens[i] = Program.library.names.index(t.name)
            
    arities         = np.array([Program.library.arities[t] for t in tokens])
    dangling        = 1 + np.cumsum(arities - 1) 
    expr_length     = 1 + np.argmax(dangling == 0)
  
    '''
        Here we return the tokens as a list of indexable integers as well as a list of library token objects. 
        We primarily need to library token objects if we want to keep track of optimized mutable constants 
    '''
    return tokens, optimized_consts, expr_length


def math_tokens_to_DEAP(tokens, primitive_set):
    """
    Transforms DSR standard tokens into DEAP format tokens.

    DSR and DEAP format are very similar, but we need to translate it over. 

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    primitive_set : gp.PrimitiveSet
        This should contain the list of primitives we will use. One way to create this is:
        
            # Create the primitive set
            pset = gp.PrimitiveSet("MAIN", dataset.X_train.shape[1])

            # Add input variables
            rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(dataset.n_input_var)}
            pset.renameArguments(**rename_kwargs)

            # Add primitives
            for k, v in function_map.items():
                if k in dataset.function_set:
                    pset.addPrimitive(v.function, v.arity, name=v.name) 

    Returns
    _______
    individual : gp.PrimitiveTree
        This is a specialized list that contains points to element from primitive_set that were mapped based 
        on the translation of the tokens. 
    """
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(tokens, np.ndarray), "Raw tokens are supplied as a numpy array."
    assert isinstance(primitive_set, gp.PrimitiveSet), "You need to supply a valid primitive set for translation."
    assert Program.library is not None, "You have to have an initial program class to supply library token conversions."
    
    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens      = _finish_tokens(tokens)
    plist       = []      
    mc_count    = 0 
    
    for t in tokens:
        
        node = Program.library[t]

        if node.name == "const":
            '''
                NUMBER - Blank floating point constant. 
                    
                    Typically this is a constant parameter we want to optimize.
            '''
            try:
                # Optimizable consts are not tracked like other terminals in DSR.
                # We just need to make sure we keep them in order. Naming is arbitrary. 
                cname   = "mutable_const_{}".format(mc_count)
                p       = primitive_set.mapping[cname]
                if node.value is not None:
                    p.value = np.float(node.value)
                else:
                    p.value = np.float(1.0)
                plist.append(p)
                mc_count += 1
            except ValueError:
                print("ERROR: Cannot add mutable \"const\" from DEAP primitve set")
                
        elif node.arity == 0 and node.input_var is None:
            '''
                NUMBER - Library supplied floating point constant. 
                    
                    This is a constant the user sets and should not change. 
            '''
            try:
                # The DSR node name is stored in the string to make it easier to map back from DEAP
                # later. 
                p       = primitive_set.mapping["user_const_{}".format(node.name)]
                p.value = node.function()
                plist.append(p)
            except ValueError:
                print("ERROR: Cannot add user \"const\" from DEAP primitve set")
                
        elif node.input_var is not None:
            '''
                NUMBER - Values from input X at location given by value in node
                
                    This is usually the raw data point numerical values. Its value should not change. 
            '''
            try:
                # Here we use x{} rather than ARG{} since we renamed it by mapping. 
                plist.append(primitive_set.mapping[node.name])
            except ValueError:
                print("ERROR: Cannot add argument value \"x{}\" from DEAP primitve set".format(node))
                
        else:
            '''
                FUNCTION - Name should map from Program. Be sure to add all function map items into PrimativeSet before call. 
                
                    This is any common function with a name like "sin" or "log". 
                    We assume right now all functions work on floating points. 
            '''
            try:
                plist.append(primitive_set.mapping[node.name])
            except ValueError:
                print("ERROR: Cannot add function \"{}\" from DEAP primitve set".format(node.name))
            
    individual = gp.PrimitiveTree(plist)
    
    '''
        Look. You've got it all wrong. You don't need to follow me. 
        You don't need to follow anybody! You've got to think for yourselves. 
        You're all individuals! 
    '''
    return individual


