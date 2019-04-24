import numpy as np

from gplearn.functions import _function_map, _Function

import utils as U


class Program():
    def __init__(self, program):
        """Build the program from a list of operators/terminals"""

        self.program = []
        count = 1
        for p in program:
            if count == 0 or p == -1:
                break
            val = U.library[p] # Value in library
            if val in U.binary + U.unary:
                self.program.append(_function_map[val.lower()])
                if val in U.binary:
                    count += 1
            elif U.library[p] in U.leaf:
                self.program.append(int(val[-1]) - 1)
                count -= 1

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


    ### Reward functions ###


    def neg_mse(self, X, y):
        '''Negative mean squared error'''

        y_hat = self.execute(X)
        return -np.mean((y - y_hat)**2)


    def fraction(self, X, y, alpha=0.25):
        '''Fraction of predicted points within alpha*y of the true value'''

        y_hat = self.execute(X)
        return np.mean(abs(y - y_hat) < alpha*abs(y))


    def __repr__(self):
        return ','.join(["x{}".format(f) if type(f) == int else f.name for f in self.program])

