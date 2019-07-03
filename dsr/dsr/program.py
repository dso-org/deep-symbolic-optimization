from textwrap import indent

import numpy as np
from numba.typed import Dict
from numba import types
from gplearn.functions import _function_map, _Function
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from dsr.const import make_const_optimizer
from dsr.utils import cached_property


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
    const_token = None      # Token corresponding to constant
    X_train = None
    y_train = None
    cache = {}


    def __init__(self, tokens, optimize):
        """
        Builds the program from a list of tokens, optimizes the constants
        against training data, and evalutes the reward.
        """

        self.traversal = [Program.library[t] for t in tokens]
        self.const_pos = [i for i,t in enumerate(tokens) if t == Program.const_token]
        if optimize:
            _, self.base_r = self.optimize()
        else:
            self.base_r = None
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
        optimized constants and base reward.

        This function generates an objective function based on the training
        dataset, reward function, and constant optimizer. It ignores penalties
        because the Program structure is fixed, thus penalties are all the same.
        It then optimizes the constants of the program and returns the optimized
        constants and base reward (reward without penalty).

        Returns
        _______
        optimized_constants : vector
            Array of optimized constants.

        base_r : float
            The base reward (reward without penalty) of the optimized program.
        """

        # Create the objective function, which is a function of the constants being optimized
        def f(consts):
            self.set_constants(consts)
            y_hat = self.execute(Program.X_train)
            obj = -1*Program.reward_function(Program.y_train, y_hat)
            return obj
        
        if len(self.const_pos) > 0:
            # Do the optimization
            x0 = np.ones(len(self.const_pos)) # Initial guess
            optimized_constants, base_r = Program.const_optimizer(f, x0)
            self.set_constants(optimized_constants)

        else:
            # No need to optimize if there are no constants
            optimized_constants = []
            base_r = -f([])

        return optimized_constants, base_r


    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            self.traversal[self.const_pos[i]] = const


    @classmethod
    def set_training_data(cls, X_train, y_train):
        """Sets the class' training data"""

        cls.X_train = X_train
        cls.y_train = y_train


    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer


    @classmethod
    def set_reward_function(cls, name, *params):
        """Sets the class' reward function"""

        all_functions = {
            # Negative mean squared error
            "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                            0),

            # Inverse mean squared error
            "inverse_mse" : (lambda y, y_hat : 1/np.mean((y - y_hat)**2),
                            0),

            # Fraction of predicted points within p0*abs(y) + p1 band of the true value
            "fraction" :    (lambda y, y_hat : np.mean(abs(y - y_hat) < params[0]*abs(y) + params[1]),
                            2),

            # Pearson correlation coefficient
            "pearson" :     (lambda y, y_hat : scipy.stats.pearsonr(y, y_hat)[0],
                            0),

            # Spearman correlation coefficient
            "spearman" :    (lambda y, y_hat : scipy.stats.spearmanr(y, y_hat)[0],
                            0)
        }

        assert name in all_functions, "Unrecognized reward function name"
        assert len(params) == all_functions[name][1], "Expected {} reward function parameters; received {}.".format(all_functions[name][1], len(params))
        Program.reward_function = all_functions[name][0]


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

        # Add input variables
        Program.library = {i : i for i in range(n_input_var)}
        Program.arities = {i : 0 for i in range(n_input_var)}
        Program.arities_numba = Dict.empty(key_type=types.int8,
                                           value_type=types.int8)
        for i in range(n_input_var):
            Program.arities_numba[i] = 0

        # Add operators
        operators = [op.lower() if isinstance(op, str) else op for op in operators]
        for i, op in enumerate(operators):

            key = i + n_input_var

            # Function
            if op in _function_map:
                op = _function_map[op]
                Program.library[key] = op
                Program.arities[key] = op.arity
                Program.arities_numba[key] = op.arity

            # Hard-coded floating-point constant
            elif isinstance(op, float):
                Program.library[key] = op
                Program.arities[key] = 0
                Program.arities_numba[key] = 0

            # Constant placeholder (to-be-optimized)
            elif op == "const":
                Program.library[key] = op
                Program.arities[key] = 0
                Program.arities_numba[key] = 0
                Program.const_token = key

            else:
                raise ValueError("Operation {} not recognized.".format(op))

        print("Library:\n\t{}".format(', '.join(["x" + str(i+1) for i in range(n_input_var)] + operators)))
        Program.library_out = [x.name if isinstance(x, _Function) else str(x) for x in Program.library]


    @staticmethod
    def convert(traversal):
        """Converts a string traversal to an int traversal"""

        str_library = [f if isinstance(f, str) else f.name for f in Program.library]
        return np.array([str_library.index(f.lower()) for f in traversal], dtype=np.int32)

    
    def reward(self, X, y):
        """Evaluates and returns the base reward under a given dataset"""

        y_hat = self.execute(X)
        return Program.reward_function(y, y_hat)


    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""

        return self.base_r - self.complexity


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
