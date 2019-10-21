from textwrap import indent

import numpy as np
from numba.typed import Dict
from numba import types
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from dsr.functions import _function_map, _Function
from dsr.const import make_const_optimizer
from dsr.utils import cached_property

import gym

"""config.json : early_stopping=false"""
env_name = "MountainCarContinuous-v0"
max_steps = 100 #in each episode
dim_of_state = 2

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

    # Truncate expressions that complete early; extend ones that don't complete
    arities = np.array([Program.arities[t] for t in tokens])
    dangling = 1 + np.cumsum(arities - 1) # Number of dangling nodes
    if 0 in dangling:
        expr_length = 1 + np.argmax(dangling == 0)
        tokens = tokens[:expr_length]
    else:
        tokens = np.append(tokens, [0]*dangling[-1]) # Extend with x1's

    # If the Program is in the cache, return it; otherwise, create a new one
    key = tokens.tostring()
    if key in Program.cache:
        p = Program.cache[key]
        p.count += 1
        return p
    else:
        p = Program(tokens, optimize=optimize)
        Program.cache[key] = p
        return p


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
        List of operators (type: _Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices 
        
    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

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
    """

    # Static variables
    library = None          # Dict of operators/terminals for each token
    arities = None          # Dict of arities for each token
    reward_function = None  # Reward function
    const_optimizer = None  # Function to optimize constants
    X_train = None
    y_train = None
    cache = {}

    # Additional derived static variables
    L = None                # Length of library    
    terminal_tokens = None  # Tokens corresponding to terminals
    unary_tokens = None     # Tokens corresponding to unary operators
    binary_tokens = None    # Tokens corresponding to binary operators
    trig_tokens = None      # Tokens corresponding to trig functions
    const_token = None      # Token corresponding to constant
    inverse_tokens = None   # Dict of token to inverse tokens
    arities_numba = None    # Numba Dict of token to arity
    parent_adjust = None    # Numba Dict to transform library key to non-terminal sub-library key



    def __init__(self, tokens, optimize):
        """
        Builds the program from a list of tokens, optimizes the constants
        against training data, and evalutes the reward.
        """

        self.traversal = [Program.library[t] for t in tokens]
        self.const_pos = [i for i,t in enumerate(tokens) if t == Program.const_token]
        if optimize:
            _ = self.optimize()
        self.count = 1


    def execute(self, X):
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

        # Check for single-node programs
        node = self.traversal[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.traversal:

            if isinstance(node, _Function):
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

        # Create the objective function, which is a function of the constants being optimized
        def f(consts):
            self.set_constants(consts)
            y_hat = self.execute(Program.X_train)
            obj = np.mean((Program.y_train - y_hat)**2)
            return obj
        
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
    def set_training_data(cls, dataset):
        """Sets the class' training and testing data"""

        cls.X_train = dataset.X_train
        cls.y_train = dataset.y_train
        cls.X_test = dataset.X_test
        cls.y_test = dataset.y_test
        cls.y_train_noiseless = dataset.y_train_noiseless
        cls.y_test_noiseless = dataset.y_test_noiseless


    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer


    
    @classmethod
    def set_reward_function(cls, name, *params):
        Program.reward_function = 0 #dummy function not used at all


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
    def set_library(cls, operators, n_input_var):
        """Sets the class library and arities"""
        """ ignore config.json set as gym """
        operators = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log'] #user defined 
        n_input_var = dim_of_state
        # Add input variables
        Program.library = {i : i for i in range(n_input_var)}
        Program.arities = {i : 0 for i in range(n_input_var)}

        # Add operators
        operators = [op.lower() if isinstance(op, str) else op for op in operators]
        for i, op in enumerate(operators):

            key = i + n_input_var

            # Function
            if op in _function_map:
                op = _function_map[op]
                Program.library[key] = op
                Program.arities[key] = op.arity

            # Hard-coded floating-point constant
            elif isinstance(op, float):
                Program.library[key] = op
                Program.arities[key] = 0

            # Constant placeholder (to-be-optimized)
            elif op == "const":
                Program.library[key] = op
                Program.arities[key] = 0
                Program.const_token = key

            else:
                raise ValueError("Operation {} not recognized.".format(op))

        # Create Numba Dicts
        Program.arities_numba = Dict.empty(key_type=types.int8, value_type=types.int8)
        for i in range(len(Program.arities)):
            Program.arities_numba[i] = Program.arities[i]

        Program.parent_adjust = Dict.empty(key_type=types.int8, value_type=types.int8)
        j = 0
        for i in range(len(Program.arities)):
            if Program.arities[i] > 0:
                Program.parent_adjust[i] = j
                j += 1

        Program.L = len(Program.library)
        trig_names = ["sin", "cos", "tan", "csc", "sec", "cot"]
        trig_names += ["arc" + name for name in trig_names]
        Program.terminal_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 0], dtype=np.int32)
        Program.unary_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 1], dtype=np.int32)
        Program.binary_tokens = np.array([t for t in range(Program.L) if Program.arities[t] == 2], dtype=np.int32)
        Program.trig_tokens = np.array([t for t in range(Program.L) if isinstance(Program.library[t], _Function) and Program.library[t].name in trig_names], dtype=np.int32)

        inverse_tokens = {
            "inv" : "inv",
            "neg" : "neg",
            "exp" : "log",
            "log" : "exp",
            "sqrt" : "n2",
            "n2" : "sqrt"
        }
        token_from_name = {v.name : k for k,v in Program.library.items() if isinstance(v, _Function)}
        Program.inverse_tokens = {token_from_name[k] : token_from_name[v] for k,v in inverse_tokens.items() if k in token_from_name and v in token_from_name}

        print("Library:\n\t{}".format(', '.join(["x" + str(i+1) for i in range(n_input_var)] + operators)))


    @staticmethod
    def convert(traversal):
        """Converts a string traversal to an int traversal"""

        str_library = [f if isinstance(f, str) else f.name for f in Program.library]
        return np.array([str_library.index(f.lower()) for f in traversal], dtype=np.int32)


    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    @cached_property
    def base_r(self):
        """GYM: Set one episdoe per one program"""
        base_reward = 0
        env = gym.envs.make(env_name)
        gym_states =  env.reset()
        for i in range(max_steps):
            env = gym.envs.make(env_name)
            gym_states =  env.reset()
            gym_action = self.execute(np.asarray([gym_states]))
            gym_states, r, done, info =  env.step(gym_action) # Do it multiples
            base_reward+=r
        base_r = base_reward    
        return base_r


    @cached_property
    def base_r_test(self):
        """Evaluates and returns the base reward of the program on the test
        set"""

        return self.base_r


    @cached_property
    def base_r_noiseless(self):
        """Evaluates and returns the base reward of the program on the noiseless
        training set"""

        return self.base_r


    @cached_property
    def base_r_test_noiseless(self):
        """Evaluates and returns the base reward of the program on the noiseless
        test set"""

        return self.base_r


    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program on the training
        set"""

        return self.base_r - self.complexity


    @cached_property
    def r_test(self):
        """Evaluates and returns the reward of the program on the test set"""

        return self.base_r_test - self.complexity


    @cached_property
    def r_noiseless(self):
        """Evaluates and returns the reward of the program on the noiseless
        training set"""

        return self.base_r_noiseless - self.complexity


    @cached_property
    def r_test_noiseless(self):
        """Evaluates and returns the reward of the program on the noiseless
        test set"""

        return self.base_r_test_noiseless - self.complexity


    @cached_property
    def nmse(self):
        """Evaluates and returns the normalized mean squared error of the
        program on the test set (used as final performance metric)"""

        y_hat = self.execute(Program.X_test)
        var_y = np.var(Program.y_test)
        return np.mean((Program.y_test - y_hat)**2) / var_y


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
        expr = parse_expr(tree.__repr__()) # SymPy expression

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

        return ','.join(["x{}".format(f + 1) if isinstance(f, int) else str(f) if isinstance(f, float) else f.name for f in self.traversal])


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

        if isinstance(op, _Function):
            val = op.name
            if val in capital:
                val = val.capitalize()
            n_children = op.arity            
        elif isinstance(op, int):
            val = "x{}".format(op + 1)
            n_children = 0
        elif isinstance(op, float):
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
