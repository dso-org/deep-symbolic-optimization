import numpy as np
from gplearn.functions import _function_map, _Function
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty


class Program():

    # Static variables
    library = None          # List of operators/terminals
    reward_function = None  # Reward function
    

    def __init__(self, tokens):
        """Build the program from a list of tokens"""

        self.program = [] # List of operators (type: _Function) and terminals (type: int, float)
        count = 1
        for t in tokens:
            if count == 0 or t == -1: # TBD: Get rid of -1 case, then move this to end of loop iteration
                break
            if isinstance(t, np.int32): # Operator or input variable
                op = Program.library[t]
                if isinstance(op, _Function):
                    count += op.arity - 1
                else:
                    op = int(op[1:])
                    count -= 1
            elif isinstance(t, np.float32): # Constant
                raise ValueError("Constants not yet supported.")
                op = float(t)
                count -= 1
            else:
                raise ValueError("Unrecognized type: {}".format(type(t)))
            self.program.append(op)

        # Complete unfinished programs with x1
        for i in range(count):
            self.program.append(0)

        self.sympy_expr = None  # Corresponding SymPy expression, only calculated for pretty print


    def execute(self, X):
        """Execute the program according to X.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

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


    @classmethod
    def set_reward_function(cls, name, *params):

        if Program.reward_function is not None:
            raise RuntimeError("Error: Cannot set reward function more than once.")


        all_functions = {
            # Negative mean squared error
            "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                            0),

            # Inverse mean squared error
            "inverse_mse" : (lambda y, y_hat : 1/np.mean((y - y_hat)**2),
                            0),

            # Fraction of predicted points within p0*abs(y) + p1 band of the true value
            "fraction" :    (lambda y, y_hat : np.mean(abs(y - y_hat) < params[0]*abs(y) + params[1]),
                            2)
        }

        assert name in all_functions, "Unrecognized reward function name"
        assert len(params) == all_functions[name][1], "Expected {} reward function parameters; received {}.".format(all_functions[name][1], len(params))
        Program.reward_function = all_functions[name][0]


    @classmethod
    def set_library(cls, operators, n_input_var):

        if Program.library is not None:
            raise RuntimeError("Error: Cannot set library more than once.")

        Program.library = []

        # Add operators
        library = [op.lower() for op in operators] # Convert to lower-case
        for op in library:
            # Function
            if op in _function_map:
                Program.library.append(_function_map[op])
            # Input variable
            elif type(op) == int:
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


    # Evaluate the reward of a given dataset
    def reward(self, X, y):
        y_hat = self.execute(X)
        return Program.reward_function(y, y_hat)


    # Get pretty printing of the program. This is actually a bit complicated because we have to go:
        # traversal --> tree --> serialized tree --> SymPy expression
    def pretty(self):
        if self.sympy_expr is None:
            tree = self.program.copy()
            tree = build_tree(tree)
            print("Before", tree.__repr__())
            tree = convert_to_sympy(tree)
            print("After ", tree.__repr__())
            self.sympy_expr = parse_expr(tree.__repr__()) # SymPy expression

        return pretty(self.sympy_expr)


    # Print the program's traversal
    def __repr__(self):
        return ','.join(["x{}".format(f + 1) if type(f) == int else f.name for f in self.program])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that SymPy capitalizes
capital = ["add", "mul", "pow"]


"""Basic tree class supporting printing"""
class Node:

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


"""Recursively builds tree from pre-order traversal"""
def build_tree(program, order="preorder"):

    if order == "preorder":
        op = program.pop(0)

        if isinstance(op, _Function):
            val = op.name
            if val in capital:
                val = val.capitalize()
            n_children = op.arity            
        elif isinstance(op, int):
            val = "x{}".format(op + 1)
            n_children = 0
        elif isinstance(op, float):
            raise ValueError("Constants not yet supported.")
            val = str(op)
            n_children = 0
        else:
            raise ValueError("Unrecognized type: {}".format(type(op)))

        node = Node(val)

        for _ in range(n_children):
            node.children.append(build_tree(program))

        return node

    elif order == "postorder":
        raise NotImplementedError

    elif order == "inorder":
        raise NotImplementedError


"""Adjusts trees to only use node values supported by SymPy"""
def convert_to_sympy(node):

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