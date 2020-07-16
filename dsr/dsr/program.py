"""Class for symbolic expression object or program."""

import array
import os
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty
import gym

from dsr.functions import function_map, Function
from dsr.const import make_const_optimizer
from dsr.utils import cached_property
import dsr.utils as U

try:
    from deap import gp 
except ImportError:
    gp = None
    
    
def _finish_tokens(tokens):
    
    arities         = np.array([Program.arities[t] for t in tokens])
    dangling        = 1 + np.cumsum(arities - 1) # Number of dangling nodes
    
    if 0 in dangling:
        expr_length     = 1 + np.argmax(dangling == 0)
        tokens          = tokens[:expr_length]
    else:
        tokens          = np.append(tokens, [0]*dangling[-1]) # Extend with x1's
    
    return tokens

def from_tokens(tokens, optimize):
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

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens = _finish_tokens(tokens)

    # For stochastic Programs, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    if Program.stochastic:
        p = Program(tokens, optimize=optimize)
    else:
        key = tokens.tostring()
        if key in Program.cache:
            p = Program.cache[key]
            p.count += 1
        else:
            p = Program(tokens, optimize=optimize)
            Program.cache[key] = p

    return p


def DEAP_to_tokens(individual):
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(individual, gp.PrimitiveTree), "Program tokens should be a Deap GP PrimativeTree object."
    
    tokens = []
    
    for t in individual:
        
        if isinstance(t, gp.Terminal):
            if t.name is "const":
                # Get the constant value as it now stands
                tokens.append(t.value)
            else:
                # Get the int which is contained in "x{}"
                tokens.append(int(t.name[1:]))
        else:
            # Get the index number for this op from the op list in Program.library
            tokens.append(Program.library.index(t.name))
        
    return tokens
        
    
def tokens_to_DEAP(tokens, primitive_set):
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
    assert isinstance(tokens, list), "Raw tokens are supplied as a list."
    assert isinstance(primitive_set, gp.PrimitiveSet), "You need to supply a valid primitive set for translation."
    assert Program.library is not None, "You have to have an initial program class to supply library token conversions."
    
    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens  = _finish_tokens(tokens)
             
    plist   = []        
    
    for t in tokens:
        
        node = Program.library[t]

        if isinstance(node, float):
            '''
                NUMBER - Library supplied floating point constant. 
                    
                    Typically this is a constant parameter we want to optimize. Its value may change. 
            '''
            try:
                p = primitive_set.context["const"]
                p.value = node
                plist.append(p)
            except ValueError:
                print("ERROR: Cannot add \"const\" from DEAP primitve set")
                
        elif isinstance(node, int):
            '''
                NUMBER - Values from input X at location given by value in node
                
                    This is usually the raw data point numerical values. Its value should not change. 
                    
            '''
            try:
                plist.append(primitive_set.context["x{}".format(node)])
            except ValueError:
                print("ERROR: Cannot add argument value \"x{}\" from DEAP primitve set".format(node))
                
        else:
            '''
                FUNCTION - Name should map from Program. Be sure to add all function map items into PrimativeSet before call. 
                
                    This is any common function with a name like "sin" or "log". 
                    We assume right now all functions work on floating points. 
            '''
            try:
                plist.append(primitive_set.context[node.name])
            except ValueError:
                print("ERROR: Cannot add function \"{}\" from DEAP primitve set".format(node.name))
                
    assert len(plist) > 1, "Single token expressions are not valid. Got {} tokens.".format(len(plist))
            
    individual = gp.PrimitiveTree(plist)
    
    '''
        Look. You've got it all wrong. You don't need to follow me. 
        You don't need to follow anybody! You've got to think for yourselves. 
        You're all individuals! 
    '''
    return individual


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
    library = None          # List of operators/terminals for each token
    arities = None          # Array of arities for each token
    reward_function = None  # Reward function
    const_optimizer = None  # Function to optimize constants
    cache = {}
    primitive_set = None
    
    # Additional derived static variables
    L = None                # Length of library
    terminal_tokens = None  # Tokens corresponding to terminals
    unary_tokens = None     # Tokens corresponding to unary operators
    binary_tokens = None    # Tokens corresponding to binary operators
    trig_tokens = None      # Tokens corresponding to trig functions
    const_token = None      # Token corresponding to constant
    inverse_tokens = None   # Dict of token to inverse tokens
    parent_adjust = None    # Array to transform library index to non-terminal sub-library index. Values of -1 correspond to invalid entry (i.e. terminal parent)

    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python
    cyfunc = None           # Link to cyfunc lib since we do an include inline
        
    def __init__(self, tokens, optimize):

        self._tokens_to_program(tokens)
        
        if optimize:
            _ = self.optimize()
        self.count = 1
    
    def _tokens_to_program(self, tokens):
        
        """
        Builds the program from a list of tokens, optimizes the constants
        against training data, and evalutes the reward.
        """
    
        self.traversal = [Program.library[t] for t in tokens]
        self.const_pos = [i for i,t in enumerate(tokens) if t == Program.const_token] # Just constant placeholder positions
            
        if self.have_cython:
            self.float_pos      = self.const_pos + [i for i,t in enumerate(tokens) if isinstance(Program.library[t], np.float32)] # Constant placeholder + floating-point positions
            self.new_traversal  = [Program.library[t] for t in tokens]
            self.is_function    = array.array('i',[isinstance(t, Function) for t in self.new_traversal])
            self.var_pos        = [i for i,t in enumerate(self.traversal) if isinstance(t, int)]   
            self.len_traversal  = len(self.traversal)
            assert self.len_traversal > 1, "Single token instances not supported"
        
        self.tokens = tokens
        self.str = tokens.tostring()
        
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
        
        return self.cyfunc.execute(X, self.len_traversal, self.traversal, self.new_traversal, self.float_pos, self.var_pos, self.is_function)
    
    
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

        # Check for single-node programs
        node = self.traversal[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.traversal:

            if isinstance(node, Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        assert False, "Function should never get here!"
        return None    
    
    
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
        # TBD: Fix to work with task refactoring

        # Create the objective function, which is a function of the constants being optimized
        def f(consts):
            self.set_constants(consts)
            y_hat = self.execute(Program.X_train)
            obj = np.mean((Program.y_train - y_hat)**2)
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
            self.traversal[self.const_pos[i]] = const


    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}


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
    def set_execute(cls):
        """Sets which execute method to use"""
        
        """
        If cython ran, we will have a 'c' file generated. The dynamic libary can be 
        given different names, so it's not reliable for testing if cython ran.
        """
        cpath = os.path.join(os.path.dirname(__file__),'cyfunc.c')
        
        if os.path.isfile(cpath):
            from .                  import cyfunc
            Program.cyfunc          = cyfunc
            Program.execute         = Program.cython_execute
            Program.have_cython     = True
        else:
            Program.execute         = Program.python_execute
            Program.have_cython     = False


    @classmethod
    def set_reward_function(cls, reward_function):
        """Sets the class reward function."""

        Program.reward_function = reward_function


    @classmethod
    def set_eval_function(cls, eval_function):
        """Sets the class eval function."""

        Program.eval_function = eval_function


    @classmethod
    def set_stochastic(cls, stochastic):
        """Sets the class stochasticity."""

        Program.stochastic = stochastic


    @classmethod
    def set_library(cls, operators, n_input_var):
        """Sets the class library and arities."""

        # Add input variables
        Program.library = list(range(n_input_var))
        Program.str_library = ["x{}".format(i+1) for i in range(n_input_var)]
        Program.arities = [0] * n_input_var

        for i, op in enumerate(operators):

            # Function
            if op in function_map:
                op = function_map[op]
                Program.library.append(op)
                Program.str_library.append(op.name)
                Program.arities.append(op.arity)

            # Hard-coded floating-point constant
            elif isinstance(op, float) or isinstance(op, int):
                op = np.float32(op)
                Program.library.append(op)
                Program.str_library.append(str(op))
                Program.arities.append(0)

            # Constant placeholder (to-be-optimized)
            elif op == "const":
                Program.library.append(op)
                Program.str_library.append(op)
                Program.arities.append(0)
                Program.const_token = i + n_input_var

            else:
                raise ValueError("Operation {} not recognized.".format(op))

        Program.arities = np.array(Program.arities, dtype=np.int32)

        count = 0
        Program.parent_adjust = np.full_like(Program.arities, -1)
        for i in range(len(Program.arities)):
            if Program.arities[i] > 0:
                Program.parent_adjust[i] = count
                count += 1

        Program.L = len(Program.library)
        trig_names = ["sin", "cos", "tan", "csc", "sec", "cot"]
        trig_names += ["arc" + name for name in trig_names]
        Program.terminal_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 0], dtype=np.int32)
        Program.unary_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 1], dtype=np.int32)
        Program.binary_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 2], dtype=np.int32)
        Program.trig_tokens = np.array([t for t in range(Program.L) if isinstance(Program.library[t], Function) and Program.library[t].name in trig_names], dtype=np.int32)

        inverse_tokens = {
            "inv" : "inv",
            "neg" : "neg",
            "exp" : "log",
            "log" : "exp",
            "sqrt" : "n2",
            "n2" : "sqrt"
        }
        token_from_name = {t.name : i for i,t in enumerate(Program.library) if isinstance(t, Function)}
        Program.inverse_tokens = {token_from_name[k] : token_from_name[v] for k,v in inverse_tokens.items() if k in token_from_name and v in token_from_name}

        print("Library:\n\t{}".format(Program.str_library))

    @classmethod
    def set_primitive_set(cls, primitive_set):
        cls.primitive_set = primitive_set

    @staticmethod
    def convert(traversal):
        """Converts a string traversal to an int traversal"""

        return np.array([Program.str_library.index(f.lower()) for f in traversal], dtype=np.int32)


    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    @cached_property
    def base_r(self):
        """Evaluates and returns the base reward of the program on the training
        set"""

        return self.reward_function()


    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program on the training
        set"""
        return self.base_r - self.complexity


    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        return self.eval_function()
    
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


    def print_stats(self):
        """Prints the statistics of the program"""
        print("\tReward: {}".format(self.r))
        print("\tBase reward: {}".format(self.base_r))
        print("\tCount: {}".format(self.count))
        print("\tTraversal: {}".format(self))
        print("\tExpression:")
        print("{}\n".format(indent(self.pretty(), '\t  ')))


    def __repr__(self):
        """Prints the program's traversal"""

        return ','.join(["x{}".format(f + 1) if isinstance(f, int) else str(f) if isinstance(f, float) or isinstance(f, np.float32) else f.name for f in self.traversal])


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


def build_tree(traversal, order="preorder"):
    """Recursively builds tree from pre-order traversal"""

    if order == "preorder":
        op = traversal.pop(0)

        if isinstance(op, Function):
            val = op.name
            if val in capital:
                val = val.capitalize()
            n_children = op.arity
        elif isinstance(op, int):
            val = "x{}".format(op + 1)
            n_children = 0
        elif isinstance(op, float) or isinstance(op, np.float32):
            val = str(op)
            n_children = 0
        else:
            raise ValueError("Unrecognized type: {}".format(type(op)))

        node = Node(val)

        for _ in range(n_children):
            node.children.append(build_tree(traversal))

        return node

    elif order == "postorder":
        raise NotImplementedError

    elif order == "inorder":
        raise NotImplementedError


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

    for child in node.children:
        convert_to_sympy(child)

    return node
