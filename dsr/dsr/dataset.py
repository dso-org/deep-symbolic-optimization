import numpy as np

from dsr.program import Program


class Dataset():

    def __init__(self,
        traversal,          # String traversal
        operators,          # Library of operators
        n_input_var,        # Number of input variables.
        train_spec,         # Training data specification
        test_spec=None,     # Testing data specification (if different)
        seed=0,             # Numpy seed used for sampling
        **kwargs
        ):
        
        self.traversal = Program.convert(traversal)
        self.program = Program(self.traversal)
        self.n_input_var = n_input_var
        self.rng = np.random.RandomState(seed)        

        self.X_train = self.make_X(train_spec)
        self.X_test = self.make_X(test_spec) if test_spec is not None else self.X_train.copy()

        self.y_train = self.program.execute(self.X_train)
        self.y_test = self.program.execute(self.X_test)


    def make_X(self, spec):
        features = []
        for i in range(1, self.n_input_var + 1):
            input_var = "x{}".format(i)
            assert input_var in spec, "No specification for input variable {}".format(input_var)
            # Format: U(low, high, n)
            if "U" in spec[input_var]:
                low, high, n = spec[input_var]["U"]
                feature = self.rng.uniform(low=low, high=high, size=n)
            # Format: E(start, stop, step) (inclusive)
            elif "E" in spec[input_var]:
                start, stop, step = spec[input_var]["E"]
                n = int((stop - start)/step)
                feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
            else:
                raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
            features.append(feature)
        
        X = np.column_stack(features)
        
        return X


