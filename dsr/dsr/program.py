"""Class for symbolic expression object or program."""

import array
import os
import warnings
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty
import gym

from dsr.functions import Token, PlaceholderConstant, function_map
from dsr.const import make_const_optimizer
from dsr.utils import cached_property
import dsr.utils as U


def _finish_tokens(tokens, n_objects: int = 1):

    """
    Finish the token strings to make sure they are a valid program. 
    
    We know we have a valid program if all arities a cancled out by 
    a the same number of terminals. Once we reach a point in the string
    where these aq equal, we cut the string of tokens. Otherwise, the 
    tokens are not yet a valid program. The solution is to keep adding 
    terminals until they fully counterweight the arities. 
    
    We have to do this since we emit programs as strings which can leave
    them over or under complete. 
    
    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. 
        
    Returns
    _______
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.
        
    """
    
    arities         = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling        = 1 + np.cumsum(arities - 1) 
    
    if -n_objects in (dangling - 1):
        # Chop off tokens once the cumsum reaches 0, This is the last valid point in the tokens
        expr_length     = 1 + np.argmax((dangling - 1) == -n_objects)
        tokens          = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))

    return tokens


def from_str_tokens(str_tokens, optimize, skip_cache=False, n_objects=1):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    optimize : bool
        See from_tokens().

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s.lower())
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                assert not optimize, "Currently does not support optimization with hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, optimize=optimize, skip_cache=skip_cache, n_objects=n_objects)

    # Replace any constants
    p.set_constants(constants)

    return p

def _set_optimized_consts(p, optimized_consts):
    """
        Sometimes things need to be imported over from places like GP/DEAP in a way that the
        standard integer tokens do not support. An example is optimized constants from a prior 
        DEAP run.
        
        Otherwise, leave it alone. 
    """
    if optimized_consts is not None:
        p.set_constants(optimized_consts)
        
    return p


def from_tokens(tokens, optimize, skip_cache=False, on_policy=True, n_objects=1, optimized_consts=None):

    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    optimize : bool
        Whether to optimize the program before returning it.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens = _finish_tokens(tokens, n_objects=n_objects)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    if skip_cache:
        p = _set_optimized_consts(Program(tokens, optimize=optimize, on_policy=on_policy, n_objects=n_objects), optimized_consts)
    elif Program.task.stochastic:
        p = _set_optimized_consts(Program(tokens, optimize=optimize, on_policy=on_policy, n_objects=n_objects), optimized_consts)
    else:
        key = tokens.tostring() 
        if key in Program.cache:
            p = Program.cache[key]
            p.count += 1
        else:
            p = _set_optimized_consts(Program(tokens, optimize=optimize, on_policy=on_policy, n_objects=n_objects), optimized_consts)
            Program.cache[key] = p

    return p


def get_best_r(p):
    """ Get the best R we already have. This is to avoid computing
        expensive validation Rs if we want to avoid it.
        
        Ww ask the local dict if we have a value in decending order
        of goodness. If we call the cached property it would compute it.
        So, we avoid that.
        
        This function is detached from the class so the cached property 
        is activated when we need it. 
    """ 
    if "long_validate" in p.__dict__:
        return p.long_validate
    elif "validate" in p.__dict__: 
        return p.validate
    else:
        return p.r


def get_best_base_r(p):
    """ Get the best base R we already have. This is to avoid computing
        expensive validation Rs if we want to avoid it.
        
        Ww ask the local dict if we have a value in decending order
        of goodness. If we call the cached property it would compute it.
        So, we avoid that.
        
        This function is detached from the class so the cached property 
        is activated when we need it.
    """ 
    if "base_long_validate" in p.__dict__:
        return p.base_long_validate
    elif "base_validate" in p.__dict__: 
        return p.base_validate
    else:
        return p.base_r


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    optimize : bool
        Whether to optimize the program upon initializing it.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    base_r : float
        The base reward (reward without penalty) of the program on the training
        data.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program on the training data.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None             # Task
    library = None          # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}

    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python
    cyfunc = None           # Link to cyfunc lib since we do an include inline

    def __init__(self, tokens, optimize, on_policy=True, n_objects=1):

        """
        Builds the Program from a list of Tokens, optimizes the Constants
        against reward function, and evalutes the reward.
        """
        
        self.traversal      = [Program.library[t] for t in tokens]
        self.const_pos      = [i for i, t in enumerate(tokens) if Program.library[t].name == "const"] # Just constant placeholder positions
        self.len_traversal  = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var    = array.array('i', [t.input_var is not None for t in self.traversal])
        
        self.invalid        = False
        self.str            = tokens.tostring()        
        self.n_objects      = n_objects
        self.tokens         = tokens
        self.do_optimize    = optimize 
        
        if optimize:
            _ = self.optimize()
            
        self.count      = 1
        self.on_policy  = on_policy # Note if a program was created on policy

        if self.n_objects > 1:
            # Fill list of multi-traversals
            #danglings = -1 * np.arange(0, self.n_objects) # dangling values to look for. When dangling (calculated below) is in this list, then an expression has ended.
            danglings = -1 * np.arange(1, self.n_objects + 1)
            self.traversals = [] # list to keep track of each multi-traversal
            i_prev = 0
            arity_list = [] # list of arities for each node in the overall traversal
            for i, token in enumerate(self.traversal):
                #import ipdb; ipdb.set_trace()
                arities = token.arity
                arity_list.append(arities)
                dangling = 1 + np.cumsum(np.array(arity_list) - 1)[-1]
                if (dangling - 1) in danglings:
                    #import ipdb; ipdb.set_trace()
                    trav_object = self.traversal[i_prev:i+1]
                    self.traversals.append(trav_object)
                    i_prev = i+1
                    """
                    Keep only what dangling values have not yet been calculated. Don't want dangling to go down and up (e.g hits -1, goes back up to 0 before hitting -2)
                    and trigger the end of a traversal at the wrong time 
                    """
                    danglings = danglings[danglings != dangling - 1]
            
    def __eq__(self, p2):
        """ Returns if the programs are the same by the string rep
        """
        return self.str == p2.str
            
    def cython_execute(self, X):
        """Executes the program according to X using Cython.

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

        if self.len_traversal > 1:
            return self.cyfunc.execute(X, self.len_traversal, self.traversal, self.is_input_var)
        else:
            return self.python_execute(X)
    
    def python_execute(self, X):
        """Executes the program according to X using Python.

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

        # # Check for single-node programs
        # node = self.traversal[0]
        # if isinstance(node, float):
        #     return np.repeat(node, X.shape[0])
        # if isinstance(node, int):
        #     return X[:, node]

        apply_stack = []

        for node in self.traversal:

            apply_stack.append([node])

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                token = apply_stack[-1][0]
                terminals = apply_stack[-1][1:]
                # terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                #              else X[:, t] if isinstance(t, int)
                #              else t for t in apply_stack[-1][1:]]
                if token.input_var is not None:
                    intermediate_result = X[:, token.input_var]
                else:
                    intermediate_result = token(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        assert False, "Function should never get here!"
        return None    
    
    def execute(self, X):
        # TODO: Consider updating execute_function to take a traversal as an argument instead of a Program. Then can clean up this function.
        # loops over n_objects and calls execute_function on each sub-traversal
        if self.n_objects > 1:
            result = []
            for i in range(self.n_objects):
                self.traversal = self.traversals[i]
                out = Program.execute_function(self, X)
                result.append(out)
            return result
        else:
            return Program.execute_function(self, X)
    
    def optimize(self):
        """
        Optimizes the constant tokens against the training data and returns the
        optimized constants.

        This function generates an objective function based on the training
        dataset, reward function, and constant optimizer. It ignores penalties
        because the Program structure is fixed, thus penalties are all the same.
        It then optimizes the constants of the program and returns the optimized
        constants.

        Returns
        _______
        optimized_constants : vector
            Array of optimized constants.
        """

        # TBD: Should return np.float32

        # Create the objective function, which is a function of the constants being optimized
        def f(consts):
            self.set_constants(consts)
            r = self.task.reward_function(self)
            obj = -r # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj
        
        assert self.execute is not None, "set_execute needs to be called first"
        
        if len(self.const_pos) > 0:
            # Do the optimization
            x0 = np.ones(len(self.const_pos)) # Initial guess
            optimized_constants = Program.const_optimizer(f, x0)
            self.set_constants(optimized_constants)

        else:
            # No need to optimize if there are no constants
            optimized_constants = []

        return optimized_constants

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            assert isinstance(const, float) or isinstance(const, np.float), "Input to program constants must be of a floating point type"
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            self.traversal[self.const_pos[i]] = PlaceholderConstant(const)


    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}


    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library


    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer


    @classmethod
    def set_complexity_penalty(cls, name, weight):
        """Sets the class' complexity penalty"""

        all_functions = {
            # No penalty
            None : lambda p : 0.0,

            # Length of tree
            "length" : lambda p : len(p)
        }

        assert name in all_functions, "Unrecognzied complexity penalty name"

        if weight == 0:
            Program.complexity_penalty = lambda p : 0.0
        else:
            Program.complexity_penalty = lambda p : weight * all_functions[name](p)


    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""
        
        """
        If cython ran, we will have a 'c' file generated. The dynamic libary can be 
        given different names, so it's not reliable for testing if cython ran.
        """
        cpath = os.path.join(os.path.dirname(__file__),'cyfunc.c')
        
        if os.path.isfile(cpath):
            from .                  import cyfunc
            Program.cyfunc          = cyfunc
            execute_function        = Program.cython_execute
            Program.have_cython     = True
        else:
            execute_function        = Program.python_execute
            Program.have_cython     = False

        if protected:
            Program.execute_function = execute_function
        else:

            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    self.error_type = None # One of ['divide', 'overflow', 'underflow', 'invalid']
                    self.error_node = None # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False # Flag for whether a warning has been encountered during a call to Program.execute()

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    if not self.new_entry: # Only record the first warning encounter
                        message = message.strip().split(' ')
                        self.error_type = message[1]
                        self.error_node = message[-1]
                    self.new_entry = True

                def update(self, p):
                    """If a floating-point error was encountered, set Program.invalid
                    to True and record the error type and error node."""

                    if self.new_entry:
                        p.invalid = True
                        p.error_type = self.error_type
                        p.error_node = self.error_node
                        self.new_entry = False


            invalid_log = InvalidLog()
            np.seterrcall(invalid_log) # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(p, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(p, X)
                    invalid_log.update(p)
                    return y

            Program.execute_function = unsafe_execute


    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    @cached_property
    def base_r(self):
        """Evaluates and returns the base reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.task.reward_function(self)

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.base_r - self.complexity

    @cached_property
    def base_validate(self):
        """Evaluates and returns the base reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.task.validate_function(self)

    @cached_property
    def validate(self):
        """Evaluates and returns the reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.base_validate - self.complexity
        
    @cached_property
    def base_long_validate(self):
        """Evaluates and returns the base reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.task.long_validate_function(self)

    @cached_property
    def long_validate(self):
        """Evaluates and returns the reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.base_long_validate - self.complexity
                
    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self.task.evaluate(self)
    
    @cached_property
    def complexity_eureqa(self):
        """Computes sum of token complexity based on Eureqa complexity measures."""

        complexity = sum([t.complexity for t in self.traversal])
        return complexity


    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        tree = self.traversal.copy()
        tree = build_tree(tree)
        tree = convert_to_sympy(tree)
        try:
            expr = parse_expr(tree.__repr__()) # SymPy expression
        except:
            expr = "N/A"
            
        return expr


    def pretty(self):
        """Returns pretty printed string of the program"""
        return pretty(self.sympy_expr)


    def print_stats(self, print_test=False):
        """Prints the statistics of the program"""
        print("\tReward: {}".format(get_best_r(self)))
        print("\tBase reward: {}".format(get_best_base_r(self)))
        if print_test:
            print("\tTest val: {}".format(self.evaluate['test_val']))
        print("\tCount: {}".format(self.count))
        print("\tInvalid: {} On Policy: {}".format(self.invalid, self.on_policy))
        print("\tTraversal: {}".format(self))
        print("\tExpression:")
        print("{}\n".format(indent(self.pretty(), '\t  ')))


    def __repr__(self):
        """Prints the program's traversal"""

        return ','.join([repr(t) for t in self.traversal])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))
        
    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))
        
    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))
        
    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))
        
    for child in node.children:
        convert_to_sympy(child)
        

        
    return node
