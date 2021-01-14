"""Class for Prior object."""

import numpy as np

from dsr.subroutines import ancestors
from dsr.library import TokenNotFoundError


def make_prior(library, config_prior):
    """Factory function for JointPrior object."""

    prior_dict = {
        "relational" : RelationalConstraint,
        "length" : LengthConstraint,
        "repeat" : RepeatConstraint,
        "inverse" : InverseUnaryConstraint,
        "trig" : TrigConstraint,
        "const" : ConstConstraint
    }

    priors = []
    warnings = []
    for prior_type, prior_args in config_prior.items():
        assert prior_type in prior_dict, \
            "Unrecognized prior type: {}.".format(prior_type)
        prior_class = prior_dict[prior_type]

        if isinstance(prior_args, dict):
            prior_args = [prior_args]
        for single_prior_args in prior_args:

            # Attempt to build the Prior. Any Prior can fail if it references a
            # Token not in the Library.
            try:
                prior = prior_class(library, **single_prior_args)
                warning = prior.validate()
            except TokenNotFoundError:
                prior = None
                warning = "Uses Tokens not in the Library."

            # Add warning context
            if warning is not None:
                warning = "Skipping invalid '{}' with arguments {}. " \
                    "Reason: {}" \
                    .format(prior_class.__name__, single_prior_args, warning)
                warnings.append(warning)

            # Add the Prior if there are no warnings
            if warning is None:
                priors.append(prior)

    joint_prior = JointPrior(library, priors)

    print("-- Building prior -------------------")
    print("\n".join(["WARNING: " + message for message in warnings]))
    print(joint_prior.describe())
    print("-------------------------------------")

    return joint_prior


class JointPrior():
    """A collection of joint Priors."""

    def __init__(self, library, priors):
        """
        Parameters
        ----------
        library : Library
            The Library associated with the Priors.

        priors : list of Prior
            The individual Priors to be joined.
        """

        self.library = library
        self.L = self.library.L
        self.priors = priors
        assert all([prior.library is library for prior in priors]), \
            "All Libraries must be identical."

        self.requires_parents_siblings = True

        self.describe()

    def initial_prior(self):
        combined_prior = np.zeros((self.L,), dtype=np.float32)
        for prior in self.priors:
            combined_prior += prior.initial_prior()
        return combined_prior

    def __call__(self, actions, parent, sibling, dangling):
        zero_prior = np.zeros((actions.shape[0], self.L), dtype=np.float32)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))]
        for i in range(len(self.priors)):
            ind_priors[i] += self.priors[i](actions, parent, sibling, dangling)
        combined_prior = sum(ind_priors) + zero_prior
        return combined_prior

    def describe(self):
        message = "\n".join(prior.describe() for prior in self.priors)
        return message


class Prior():
    """Abstract class whose call method return logits."""

    def __init__(self, library):
        self.library = library
        self.L = library.L

    def validate(self):
        """
        Determine whether the Prior has a valid configuration. This is useful
        when other algorithmic parameters may render the Prior degenerate. For
        example, having a TrigConstraint with no trig Tokens.

        Returns
        -------
        message : str or None
            Error message if Prior is invalid, or None if it is valid.
        """

        return None

    def init_zeros(self, actions):
        """Helper function to generate a starting prior of zeros."""

        batch_size = actions.shape[0]
        prior = np.zeros((batch_size, self.L), dtype=np.float32)
        return prior

    def initial_prior(self):
        """
        Compute the initial prior, before any actions are selected.

        Returns
        -------
        initial_prior : array
            Initial logit adjustment before actions are selected. Shape is
            (self.L,) as it will be broadcast to batch size later.
        """

        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):
        """
        Compute the prior (logit adjustment) given the current actions.

        Returns
        -------
        prior : array
            Logit adjustment for selecting next action. Shape is (batch_size,
            self.L).
        """

        raise NotImplementedError

    def describe(self):
        """Describe the Prior."""

        message = "No description."
        return message


class Constraint(Prior):
    def __init__(self, library):
        Prior.__init__(self, library)

    def make_constraint(self, mask, tokens):
        """
        Generate the prior for a batch of constraints and the corresponding
        Tokens to constrain.

        For example, with L=5 and tokens=[1,2], a constrained row of the prior
        will be: [0.0, -np.inf, -np.inf, 0.0, 0.0].

        Parameters
        __________

        mask : np.ndarray, shape=(?,), dtype=np.bool_
            Boolean mask of samples to constrain.

        tokens : np.ndarray, dtype=np.int32
            Tokens to constrain.

        Returns
        _______

        prior : np.ndarray, shape=(?, L), dtype=np.float32
            Logit adjustment. Since these are hard constraints, each element is
            either 0.0 or -np.inf.
        """

        prior = np.zeros((mask.shape[0], self.L), dtype=np.float32)
        for t in tokens:
            prior[mask, t] = -np.inf
        return prior


class RelationalConstraint(Constraint):
    """
    Class that constrains the following:

        Constrain (any of) `targets` from being the `relationship` of (any of)
        `effectors`.

    Parameters
    ----------
    targets : list of Tokens
        List of Tokens, all of which will be constrained if any of effectors
        are the given relationship.

    effectors : list of Tokens
        List of Tokens, any of which will cause all targets to be constrained
        if they are the given relationship.

    relationship : choice of ["child", "descendant", "sibling", "uchild"]
        The type of relationship to constrain.
    """

    def __init__(self, library, targets, effectors, relationship):
        Prior.__init__(self, library)
        self.targets = library.actionize(targets)
        self.effectors = library.actionize(effectors)
        self.relationship = relationship

    def validate(self):
        message = []
        if self.relationship in ["child", "descendant", "uchild"]:
            if np.isin(self.effectors, self.library.terminal_tokens).any():
                message = "{} relationship cannot have terminal effectors." \
                          .format(self.relationship.capitalize())
                return message
        if len(self.targets) == 0:
            message = "There are no target Tokens."
            return message
        if len(self.effectors) == 0:
            message = "There are no effector Tokens."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):

        if self.relationship == "descendant":
            mask = ancestors(actions=actions,
                             arities=self.library.arities,
                             ancestor_tokens=self.effectors)
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "child":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.isin(parent, adj_parents)
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "sibling":
            # The sibling relationship is reflexive: if A is a sibling of B,
            # then B is also a sibling of A. Thus, we combine two priors, where
            # targets and effectors are swapped.
            mask = np.isin(sibling, self.effectors)
            prior = self.make_constraint(mask, self.targets)
            mask = np.isin(sibling, self.targets)
            prior += self.make_constraint(mask, self.effectors)

        elif self.relationship == "uchild":
            # Case 1: parent is a unary effector
            unary_effectors = np.intersect1d(self.effectors,
                                             self.library.unary_tokens)
            adj_unary_effectors = self.library.parent_adjust[unary_effectors]
            mask = np.isin(parent, adj_unary_effectors)
            # Case 2: sibling is a target and parent is an effector
            adj_effectors = self.library.parent_adjust[self.effectors]
            mask += np.logical_and(np.isin(sibling, self.targets),
                                   np.isin(parent, adj_effectors))
            prior = self.make_constraint(mask, [self.targets])

        return prior

    def describe(self):

        targets = ", ".join([self.library.names[t] for t in self.targets])
        effectors = ", ".join([self.library.names[t] for t in self.effectors])
        relationship = {
            "child" : "a child",
            "sibling" : "a sibling",
            "descendant" : "a descendant",
            "uchild" : "the only unique child"
        }[self.relationship]
        message = "[{}] cannot be {} of [{}]." \
                  .format(targets, relationship, effectors)
        return message


class TrigConstraint(RelationalConstraint):
    """Class that constrains trig Tokens from being the desendants of trig
    Tokens."""

    def __init__(self, library):
        targets = library.trig_tokens
        effectors = library.trig_tokens
        RelationalConstraint.__init__(self, library,
                                      targets=targets,
                                      effectors=effectors,
                                      relationship="descendant")


class ConstConstraint(RelationalConstraint):
    """Class that constrains the const Token from being the only unique child
    of all non-terminal Tokens."""

    def __init__(self, library):
        targets = library.const_token
        effectors = np.concatenate([library.unary_tokens,
                                    library.binary_tokens])
        RelationalConstraint.__init__(self, library,
                                      targets=targets,
                                      effectors=effectors,
                                      relationship="uchild")


class InverseUnaryConstraint(Constraint):
    """Class that constrains each unary Token from being the child of its
    corresponding inverse unary Tokens."""

    def __init__(self, library):
        Prior.__init__(self, library)
        self.priors = []
        for target, effector in library.inverse_tokens.items():
            targets = [target]
            effectors = [effector]
            prior = RelationalConstraint(library,
                                         targets=targets,
                                         effectors=effectors,
                                         relationship="child")
            self.priors.append(prior)

    def validate(self):
        if len(self.priors) == 0:
            message = "There are no inverse unary Token pairs in the Library."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):
        prior = sum([prior(actions, parent, sibling, dangling)
                     for prior in self.priors])
        return prior

    def describe(self):
        message = [prior.describe() for prior in self.priors]
        return "\n".join(message)


class RepeatConstraint(Constraint):
    """Class that constrains Tokens to appear between a minimum and/or maximum
    number of times."""

    def __init__(self, library, tokens, min_=None, max_=None):
        """
        Parameters
        ----------
        tokens : Token or list of Tokens
            Token(s) which should, in total, occur between min_ and max_ times.

        min_ : int or None
            Minimum number of times tokens should occur.

        max_ : int or None
            Maximum number of times tokens should occur.
        """

        Prior.__init__(self, library)
        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."
        self.min = min_
        self.max = max_
        self.tokens = library.actionize(tokens)

        assert min_ is None, "Repeat minimum constraints are not yet " \
            "supported. This requires knowledge of length constraints."

    def __call__(self, actions, parent, sibling, dangling):
        counts = np.sum(np.isin(actions, self.tokens), axis=1)
        prior = self.init_zeros(actions)
        if self.min is not None:
            raise NotImplementedError
        if self.max is not None:
            mask = counts >= self.max
            prior += self.make_constraint(mask, self.tokens)
        return prior

    def describe(self):
        names = ", ".join([self.library.names[t] for t in self.tokens])
        if self.min is None:
            message = "[{}] cannot occur more than {} times."\
                .format(names, self.max)
        elif self.max is None:
            message = "[{}] must occur at least {} times."\
                .format(names, self.min)
        else:
            message = "[{}] must occur between {} and {} times."\
                .format(names, self.min, self.max)
        return message


class LengthConstraint(Constraint):
    """Class that constrains the Program from falling within a minimum and/or
    maximum length"""

    def __init__(self, library, min_=None, max_=None):
        """
        Parameters
        ----------
        min_ : int or None
            Minimum length of the Program.

        max_ : int or None
            Maximum length of the Program.
        """

        Prior.__init__(self, library)
        self.min = min_
        self.max = max_

        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."

    def initial_prior(self):
        prior = Prior.initial_prior(self)
        for t in self.library.terminal_tokens:
            prior[t] = -np.inf
        return prior

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        i = actions.shape[1] - 1 # Current time

        # Never need to constrain max length for first half of expression
        if self.max is not None and (i + 2) >= self.max // 2:
            remaining = self.max - (i + 1)
            # assert sum(dangling > remaining) == 0, (dangling, remaining)
            mask = dangling >= remaining - 1 # Constrain binary
            prior += self.make_constraint(mask, self.library.binary_tokens)
            mask = dangling == remaining # Constrain unary
            prior += self.make_constraint(mask, self.library.unary_tokens)

        # Constrain terminals when dangling == 1 until selecting the
        # (min_length)th token
        if self.min is not None and (i + 2) < self.min:
            mask = dangling == 1 # Constrain terminals
            prior += self.make_constraint(mask, self.library.terminal_tokens)

        return prior

    def describe(self):
        message = []
        if self.min is not None:
            message.append("Sequences have minimum length {}.".format(self.min))
        if self.max is not None:
            message.append("Sequences have maximum length {}.".format(self.max))
        message = "\n".join(message)
        return message


class UniformArityPrior(Prior):
    """Class that puts a fixed prior on arities by transforming the initial
    distribution from uniform over tokens to uniform over arities."""

    def __init__(self, library):

        Prior.__init__(self, library)

        # For each token, subtract log(n), where n is the total number of tokens
        # in the library with the same arity as that token. This is equivalent
        # to... For each arity, subtract log(n) from tokens of that arity, where
        # n is the total number of tokens of that arity
        self.logit_adjust = np.zeros((self.L,), dtype=np.float32)
        for arity, tokens in self.library.tokens_of_arity.items():
            self.logit_adjust[tokens] -= np.log(len(tokens))

    def initial_prior(self):
        return self.logit_adjust

    def __call__(self, actions, parent, sibling, dangling):

        # This will be broadcast when added to the joint prior
        prior = self.logit_adjust
        return prior


class SoftLengthPrior(Prior):
    """Class the puts a soft prior on length. Before loc, terminal probabilities
    are scaled by exp(-(t - loc) ** 2 / (2 * scale)) where dangling == 1. After
    loc, non-terminal probabilities are scaled by that number."""

    def __init__(self, library, loc, scale):

        Prior.__init__(self, library)

        self.loc = loc
        self.scale = scale

        self.terminal_mask = np.zeros((self.L,), dtype=np.bool)
        self.terminal_mask[self.library.terminal_tokens] = True

        self.nonterminal_mask = ~self.terminal_mask

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        t = actions.shape[1] # Current time

        # Adjustment to terminal or non-terminal logits
        logit_adjust = -(t - self.loc) ** 2 / (2 * self.scale)

        # Before loc, decrease p(terminal) where dangling == 1
        if t < self.loc:
            prior[dangling == 1] += self.terminal_mask * logit_adjust

        # After loc, decrease p(non-terminal)
        else:
            prior += self.nonterminal_mask * logit_adjust

        return prior
