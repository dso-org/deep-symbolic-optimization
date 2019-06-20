import numpy as np
from gplearn.functions import _function_map, _Function
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from dsr.const import make_const_optimizer
from dsr.utils import cached_property


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises "tokens" that correspond to unary/binary operators,
    constant placeholder (to-be-optimized), input variables, or hard-coded
    constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: _Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.
        
    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    sympy_expr : str or None
        The (lazily calculated) sympy expression corresponding to the program.
        Used for pretty printing _only_.

    base_r : float or None
        The (lazily calculated) base reward (reward without penalty) of the
        program on the training data.

    complexity : float
        The (lazily calcualted) complexity of the program.
    """

    # Static variables
    library = None          # List of operators/terminals
    reward_function = None  # Reward function
    const_optimizer = None  # Function to optimize constants
    

    def __init__(self, tokens):
        """Builds the program from a list of tokens."""

        self.traversal = []               # List of operators (type: _Function) and terminals (type: int, float, str ("const"))
        self.const_pos = []             # Indices of constant tokens
        count = 1
        for i, t in enumerate(tokens):
            if count == 0 or t == -1: # TBD: Get rid of -1 case, then move this to end of loop iteration
                break
            op = Program.library[t]
            if isinstance(op, _Function):
                count += op.arity - 1
            elif op == "const":
                count -= 1
                self.const_pos.append(i)
            elif isinstance(op, str):
                op = int(op[1:])
                count -= 1
            elif isinstance(op, float):
                count -= 1
            else:
                raise ValueError("Unrecognized type: {}".format(type(op)))
            self.traversal.append(op)

        # Complete unfinished traversals with x1
        for i in range(count):
            self.traversal.append(0)

        self.sympy_expr = None # Corresponding sympy expression, only calculated for pretty print
        self.base_r = None


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

    
    def optimize(self, X, y):
        """
        Optimizes the constant tokens against a dataset.

        This function generates an objective function based on the training
        dataset, reward function, and constant optimizer. It ignores penalties
        because the Program structure is fixed, thus penalties are all the same.
        It then optimizes the constants of the program. Since reward for the
        optimized constants is already computed, this function also sets
        self.base_r.

        Parameters
        ----------
        X, y : np.ndarray
            Training data used for optimization.

        Returns
        -------

        self : Program
            Returns self with optimized constants replaced in self.traversal.
        """

        # Create the objective function, which is a function of the constants being optimized
        def f(consts):
            self.set_constants(consts)                  # Set the constants
            y_hat = self.execute(X)                     # Compute predicted values
            obj = -1*Program.reward_function(y, y_hat)  # Compute the objective
            return obj
        
        if len(self.const_pos) > 0:
            # Do the optimization
            x0 = np.ones(len(self.const_pos)) # Initial guess
            x, base_r = Program.const_optimizer(f, x0)

            # Set the optimized constants
            self.set_constants(x)            

        else:
            # No need to optimize if there are no constants
            base_r = -f([])

        self.base_r = base_r

        return self


    def set_constants(self, consts):
        """Sets the program's constant values"""

        for i, const in enumerate(consts):
            self.traversal[self.const_pos[i]] = const


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
        """Sets the class library"""

        Program.library = []

        # Add operators
        operators = [op.lower() if isinstance(op, str) else op for op in operators] # Convert strings to lower-case
        for op in operators:
            # Function
            if op in _function_map:
                Program.library.append(_function_map[op])

            # Input variable
            elif type(op) == int:
                Program.library.append(op)

            # Hard-coded floating-point constant
            elif isinstance(op, float):
                Program.library.append(op)

            # Constant placeholder (to-be-optimized)
            elif op == "const":
                Program.library.append(op)

            else:
                raise ValueError("Operation {} not recognized.".format(op))

        # Add input variables
        input_vars = ["x{}".format(i) for i in range(n_input_var)] # x0, x1, ..., x{n-1}
        Program.library.extend(input_vars)

        print("Library:\n\t{}".format([x.name if isinstance(x, _Function) else str(x) for x in Program.library]))


    @staticmethod
    def convert(traversal):
        """Converts a string traversal to an int traversal"""

        str_library = [f if isinstance(f, str) else f.name for f in Program.library]
        return np.array([str_library.index(f.lower()) for f in traversal], dtype=np.int32)

    
    def reward(self, X, y):
        """Evaluates and returns the reward of a given dataset"""

        y_hat = self.execute(X)
        return Program.reward_function(y, y_hat)


    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    def get_sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> sympy expression
        """

        if self.sympy_expr is None:
            tree = self.traversal.copy()
            tree = build_tree(tree)
            tree = convert_to_sympy(tree)
            self.sympy_expr = parse_expr(tree.__repr__()) # sympy expression

        return self.sympy_expr


    def pretty(self):
        """Returns pretty printed string of the program"""

        return pretty(self.get_sympy_expr())

    
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