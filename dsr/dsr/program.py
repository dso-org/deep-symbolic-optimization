import numpy as np
from gplearn.functions import _function_map, _Function


class Program():

    # Static variables
    library = None          # List of operators/terminals
    reward_function = None  # Reward function

    def __init__(self, program):
        """Build the program from a list of tokens"""

        self.program = [] # List of operators/terminals
        count = 1
        for p in program:
            if count == 0 or p == -1: # TBD: Get rid of -1 case, then move this to end of loop iteration
                break
            if isinstance(p, np.int32): # Operator or input variable
                op = Program.library[p]
                if isinstance(op, _Function):
                    self.program.append(op)
                    count += op.arity - 1
                else:
                    self.program.append(int(op[1:]))
                    count -= 1
            elif isinstance(p, np.float32): # Constant
                raise ValueError("Constants not yet supported.")
                self.program.append(p)
                count -= 1
            else:
                raise ValueError("Unrecognized type")


        # Complete unfinished programs with x1
        for i in range(count):
            self.program.append(0)


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


    @staticmethod
    def convert(traversal):
        """Converts a string traversal to an int traversal"""

        str_library = [f if isinstance(f, str) else f.name for f in Program.library]
        return np.array([str_library.index(f.lower()) for f in traversal], dtype=np.int32)


    # Evaluate the reward of a given dataset
    def reward(self, X, y):
        y_hat = self.execute(X)
        return Program.reward_function(y, y_hat)


    def __repr__(self):
        return ','.join(["x{}".format(f) if type(f) == int else f.name for f in self.program])

