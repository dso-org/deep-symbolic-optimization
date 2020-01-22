import numpy as np
import array
from dsr.functions import _function_map, _Function

# Cython specific C imports
cimport numpy as np
from cpython cimport array
cimport cython
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject

# Static inits
cdef list apply_stack   = [[None for i in range(25)] for i in range(1024)]
cdef int *stack_count   = <int *> malloc(1024 * sizeof(int))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function  
def execute(np.ndarray X, int len_traversal, list traversal, list new_traversal, list const_pos, list int_pos):
            
    """Executes the program according to X.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    
    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """
    #sp              = 0 # allow a dummy first row, requires a none type function with arity of -1
    
    cdef int        sp              = -1
    cdef int        Xs              = X.shape[0]
        
    cdef int        arity
    cdef np.ndarray intermediate_result
    cdef PyObject*  node

    # OTHER VARIABLES TO BIND
    ##stack_end
    ##stack_end_function
    ##_Function
        
    for n in const_pos:
        new_traversal[n] = np.repeat(traversal[n], Xs)
        
    for n in int_pos:
        new_traversal[n] = X[:, traversal[n]] 
                   
    for i in range(len_traversal):
        
        node = <PyObject*>new_traversal[i]

        if isinstance(<object>node, _Function): # GET RID OF ISINSTANCE
            sp += 1
            # Move this to the front with a memset call
            stack_count[sp]                     = 0
            # Store the reference to stack_count[sp] rather than keep calling
            apply_stack[sp][stack_count[sp]]    = <object>node
            stack_end                           = apply_stack[sp]
            # The first element is the function itself
            stack_end_function                  = stack_end[0]
            arity                               = stack_end_function.arity
        else:
            # Not a function, so lazily evaluate later
            stack_count[sp] += 1
            stack_end[stack_count[sp]]          = <object>node

        # Keep on doing this so long as arity matches up, we can 
        # add in numbers above and complete the arity later.
        while stack_count[sp] == arity:
            intermediate_result = stack_end_function(*stack_end[1:(stack_count[sp] + 1)])

            # I think we can get rid of this line ...
            if sp == 0:
                return intermediate_result
            
            sp -= 1
            # Adjust pointer at the end of the stack
            stack_end                   = apply_stack[sp]
            stack_count[sp] += 1
            stack_end[stack_count[sp]]  = intermediate_result
            # The first element is the function itself
            stack_end_function          = stack_end[0]
            arity                       = stack_end_function.arity
      
    # We should never get here
    assert False, "Function should never get here!"
    return None