"""Class for Prior object."""

import numpy as np

from dsr.subroutines import ancestors


def make_prior(library, config_prior):
    """Factory function for JointPrior object."""

    prior_dict = {
        "relational" : RelationalConstraint,
        "length" : LengthConstraint,
        "repeat" : RepeatConstraint,
        "inverse" : InverseUnaryConstraint,
        "trig" : TrigConstraint
    }

    priors = []
    for prior_type, prior_args in config_prior.items():
        assert prior_type in prior_dict, \
            "Unrecognized prior type: {}".format(prior_type)
        prior = prior_dict[prior_type](library, **prior_args)
        priors.append(prior)

    joint_prior = JointPrior(library, priors)
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

        # TBD: Determine
        self.requires_parents_siblings = True

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
        combined_prior = sum(ind_priors) + zero_prior # TBD FIX HACK
        # TBD: Status report if any samples have no choices
        return combined_prior


class Prior():
    """Abstract class whose call method return logits."""

    def __init__(self, library):
        self.library = library
        self.L = library.L

    def zeros(self, actions):
        """Helper function to generate the starting prior."""

        batch_size = actions.shape[0]
        prior = np.zeros((batch_size, self.L), dtype=np.float32)
        return prior

    def initial_prior(self):
        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):
        raise NotImplementedError


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
    Class that adds the following constraint:

        Constrain (all of) `targets` from being the `relationship` of (any of)
        `effectors`.

    Parameters
    ----------
    targets : list of Tokens
        List of Tokens, all of which will be constrained if any of effectors
        are the given relationship.

    effectors : list of Tokens
        List of Tokens, any of which will cause all constrained_tokens to be
        constrained if they are the given relationship.

    relationship : choice of ["child", "descendant", "sibling"]
        The type of relationship to constrain.
    """

    def __init__(self, library, targets, effectors, relationship):
        Prior.__init__(self, library)
        self.targets = library.actionize(targets)
        self.effectors = library.actionize(effectors)
        self.relationship = relationship

        # TBD assertions

    def __call__(self, actions, parent, sibling, dangling):

        # The ancestors subroutine already loops over effectors
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

        # The sibling relationship is reflexive: if A is a sibling of B, then B
        # is also a sibling of A. Thus, we combine two priors, where targets
        # and effectors are swapped.
        elif self.relationship == "sibling":
            mask = np.isin(sibling, self.effectors)
            prior = self.make_constraint(mask, self.targets)

            mask = np.isin(sibling, self.targets)
            prior += self.make_constraint(mask, self.effectors)

        return prior


class TrigConstraint(RelationalConstraint):
    """Class that constrains trig Tokens from being the desendants of trig
    Tokens."""

    def __init__(self, library):
        targets = library.trig_tokens
        relationship = "descendant"
        effectors = library.trig_tokens
        RelationalConstraint.__init__(self, library,
                                      targets=targets,
                                      effectors=effectors,
                                      relationship=relationship)


class InverseUnaryConstraint(RelationalConstraint):
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

    def __call__(self, actions, parent, sibling, dangling):
        prior = sum([prior(actions, parent, sibling, dangling)
                     for prior in self.priors])
        return prior

        # targets = list(library.inverse_tokens.keys())
        # relationship = "child"
        # effectors = list(library.inverse_tokens.values())
        # RelationalConstraint.__init__(self, library,
        #                               targets=targets,
        #                               effectors=effectors,
        #                               relationship=relationship)


class RepeatConstraint(Constraint):
    """Class that constrains a particular Token to appear between a minimum
    and/or maximum number of times"""

    def __init__(self, library, tokens, min_=None, max_=None):
        """
        Parameters
        ----------
        tokens : Token or list of Tokens
            Token(s) which should each appear between min_ and max_ times.

        min_ : int or None
            Minimum length of the Program.

        max_ : int or None
            Maximum length of the Program.
        """

        Prior.__init__(self, library)
        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."
        self.min = min_
        self.max = max_
        self.tokens = list(tokens)

    def __call__(self, actions, parent, sibling, dangling):
        raise NotImplementedError


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
        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."
        self.min = min_
        self.max = max_

    def initial_prior(self):
        prior = Prior.initial_prior(self)
        for t in self.library.terminal_tokens:
            prior[t] = -np.inf
        return prior

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.zeros(actions)
        i = actions.shape[1] - 1 # Current time

        # Never need to constrain max length for first half of expression
        if self.max is not None and (i + 2) >= self.max // 2:
            remaining = self.max - (i + 1)
            # assert sum(dangling > remaining) == 0, (dangling, remaining)
            # TBD: For loop over arities
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
