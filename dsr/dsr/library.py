"""Classes for Token and Library"""

from collections import defaultdict

import numpy as np


class Token():
    """
    An arbitrary token or "building block" of a Program object.

    Attributes
    ----------
    name : str
        Name of token.

    arity : int
        Arity (number of arguments) of token.

    complexity : float
        Complexity of token.

    function : callable
        Function associated with the token; used for exectuable Programs.

    input_var : int or None
        Index of input if this Token is an input variable, otherwise None.

    Methods
    -------
    __call__(input)
        Call the Token's function according to input.
    """

    def __init__(self, function, name, arity, complexity, input_var=None):
        self.function = function
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

        if input_var is not None:
            assert function is None, "Input variables should not have functions."
            assert arity == 0, "Input variables should have arity zero."

    def __call__(self, *args):
        assert self.function is not None, \
            "Token {} is not callable.".format(self.name)

        return self.function(*args)

    def __repr__(self):
        return self.name


class PlaceholderConstant(Token):
    """
    A Token for placeholder constants that will be optimized with respect to
    the reward function. The function simply returns the "value" attribute.

    Parameters
    ----------
    value : float or None
        Current value of the constant, or None if not yet set.
    """

    def __init__(self, value=None):
        if value is not None:
            value = np.atleast_1d(value)
        self.value = value

        def function():
            assert self.value is not None, \
                "Constant is not callable with value None."
            return self.value

        super().__init__(function=function, name="const", arity=0, complexity=1)

    def __repr__(self):
        if self.value is None:
            return self.name
        return str(self.value[0])


class Library():
    """
    Library of Tokens. We use a list of Tokens (instead of set or dict) since
    we so often index by integers given by the Controller.

    Attributes
    ----------
    tokens : list of Token
        List of available Tokens in the library.

    names : list of str
        Names corresponding to Tokens in the library.

    arities : list of int
        Arities corresponding to Tokens in the library.
    """

    def __init__(self, tokens):

        self.tokens = tokens
        self.L = len(tokens)
        self.names = [t.name for t in tokens]
        self.arities = np.array([t.arity for t in tokens], dtype=np.int32)

        self.input_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.input_var is not None],
            dtype=np.int32)

        def get_tokens_of_arity(arity):
            _tokens = [i for i in range(self.L) if self.arities[i] == arity]
            return np.array(_tokens, dtype=np.int32)

        self.tokens_of_arity = defaultdict(lambda : np.array([], dtype=np.int32))
        for arity in self.arities:
            self.tokens_of_arity[arity] = get_tokens_of_arity(arity)
        self.terminal_tokens = self.tokens_of_arity[0]
        self.unary_tokens = self.tokens_of_arity[1]
        self.binary_tokens = self.tokens_of_arity[2]

        try:
            self.const_token = self.names.index("const")
        except ValueError:
            self.const_token = None
        self.parent_adjust = np.full_like(self.arities, -1)
        count = 0
        for i in range(len(self.arities)):
            if self.arities[i] > 0:
                self.parent_adjust[i] = count
                count += 1

        trig_names = ["sin", "cos", "tan", "csc", "sec", "cot"]
        trig_names += ["arc" + name for name in trig_names]

        self.float_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.arity == 0 and t.input_var is None],
            dtype=np.int32)
        self.trig_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.name in trig_names],
            dtype=np.int32)

        inverse_tokens = {
            "inv" : "inv",
            "neg" : "neg",
            "exp" : "log",
            "log" : "exp",
            "sqrt" : "n2",
            "n2" : "sqrt"
        }
        token_from_name = {t.name : i for i, t in enumerate(self.tokens)}
        self.inverse_tokens = {token_from_name[k] : token_from_name[v] for k, v in inverse_tokens.items() if k in token_from_name and v in token_from_name}        

    def __getitem__(self, val):
        """Shortcut to get Token by name or index."""

        if isinstance(val, str):
            try:
                i = self.names.index(val)
            except ValueError:
                raise TokenNotFoundError("Token {} does not exist.".format(val))
        elif isinstance(val, (int, np.integer)):
            i = val
        else:
            raise TokenNotFoundError("Library must be indexed by str or int, not {}.".format(type(val)))

        try:
            token = self.tokens[i]
        except IndexError:
            raise TokenNotFoundError("Token index {} does not exist".format(i))
        return token

    def tokenize(self, inputs):
        """Convert inputs to list of Tokens."""

        if isinstance(inputs, str):
            inputs = inputs.split(',')
        elif not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            inputs = [inputs]
        tokens = [input_ if isinstance(input_, Token) else self[input_] for input_ in inputs]
        return tokens

    def actionize(self, inputs):
        """Convert inputs to array of 'actions', i.e. ints corresponding to
        Tokens in the Library."""

        tokens = self.tokenize(inputs)
        actions = np.array([self.tokens.index(t) for t in tokens],
                           dtype=np.int32)
        return actions


class TokenNotFoundError(Exception):
    pass
