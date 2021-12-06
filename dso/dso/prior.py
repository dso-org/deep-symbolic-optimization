"""Class for Prior object."""
import numpy as np
import warnings
import inspect
from collections import defaultdict

from dso.library import TokenNotFoundError
from dso.subroutines import ancestors
from dso.subroutines import jit_check_constraint_violation, \
        jit_check_constraint_violation_descendant_with_target_tokens, \
        jit_check_constraint_violation_descendant_no_target_tokens, \
        jit_check_constraint_violation_uchild, get_position, get_mask
from dso.program import Program
from dso.language_model import LanguageModelPrior as LM
from dso.utils import import_custom_source


def make_prior(library, config_prior):
    """Factory function for JointPrior object."""

    prior_dict = {
        "relational" : RelationalConstraint,
        "length" : LengthConstraint,
        "repeat" : RepeatConstraint,
        "inverse" : InverseUnaryConstraint,
        "trig" : TrigConstraint,
        "const" : ConstConstraint,
        "no_inputs" : NoInputsConstraint,
        "soft_length" : SoftLengthPrior,
        "uniform_arity" : UniformArityPrior,
        "language_model" : LanguageModelPrior
    }

    count_constraints = config_prior.pop("count_constraints", False)

    priors = []
    warn_messages = []
    for prior_type, prior_args in config_prior.items():
        if prior_type in prior_dict:
            prior_class = prior_dict[prior_type]
        else:
            # Tries to import custom priors
            prior_class = import_custom_source(prior_type)

        if isinstance(prior_args, dict):
            prior_args = [prior_args]
        for single_prior_args in prior_args:
            # Attempt to build the Prior. Any Prior can fail if it references a
            # Token not in the Library.
            prior_is_enabled = single_prior_args.pop('on', False)
            if prior_is_enabled:
                try:
                    prior = prior_class(library, **single_prior_args)
                    warn_message = prior.validate()
                except TokenNotFoundError:
                    prior = None
                    warn_message = "Uses Tokens not in the Library."
            else:
                prior = None
                warn_message = "Prior disabled."

            # Add warning context
            if warn_message is not None:
                warn_message = "Skipping invalid '{}' with arguments {}. " \
                    "Reason: {}" \
                    .format(prior_class.__name__, single_prior_args, warn_message)
                warn_messages.append(warn_message)

            # Add the Prior if there are no warnings
            if warn_message is None:
                priors.append(prior)

    joint_prior = JointPrior(library, priors, count_constraints)

    print("-- BUILDING PRIOR START -------------")
    print("\n".join(["WARNING: " + message for message in warn_messages]))
    print(joint_prior.describe())
    print("-- BUILDING PRIOR END ---------------\n")

    return joint_prior


class JointPrior():
    """A collection of joint Priors."""

    def __init__(self, library, priors, count_constraints=False):
        """
        Parameters
        ----------
        library : Library
            The Library associated with the Priors.

        priors : list of Prior
            The individual Priors to be joined.

        count_constraints : bool
            Whether to count the number of constrained tokens.
        """

        self.library = library
        self.L = self.library.L
        self.priors = priors
        assert all([prior.library is library for prior in priors]), \
            "All Libraries must be identical."

        # Name the priors, e.g. RepeatConstraint-0
        counts = defaultdict(lambda : -1)
        self.names = []
        for prior in self.priors:
            name = prior.__class__.__name__
            counts[name] += 1
            self.names.append("{}-{}".format(name, counts[name]))

        # Initialize variables for constraint count report
        self.do_count = count_constraints
        self.constraint_indices = [i for i, prior in enumerate(self.priors) if isinstance(prior, Constraint)]
        self.constraint_counts = [0] * len(self.constraint_indices)
        self.total_constraints = 0
        self.total_tokens = 0

        self.requires_parents_siblings = True 

        self.describe()

    def initial_prior(self):
        combined_prior = np.zeros((self.L,), dtype=np.float32)
        for prior in self.priors:
            combined_prior += prior.initial_prior()
        return combined_prior

    def __call__(self, actions, parent, sibling, dangling):
        # Sum the individual priors
        zero_prior = np.zeros((actions.shape[0], self.L), dtype=np.float32)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))]
        for i in range(len(self.priors)):
            ind_priors[i] += self.priors[i](actions, parent, sibling, dangling)
        combined_prior = sum(ind_priors) + zero_prior 

        # Count number of constrained tokens per prior
        if self.do_count:
            mask = dangling > 0 
            self.total_tokens += mask.sum() * actions.shape[1]
            for i in self.constraint_indices:
                self.constraint_counts[i] += np.count_nonzero(ind_priors[i][mask] == -np.inf)
            self.total_constraints += np.count_nonzero(combined_prior[mask] == -np.inf)

        return combined_prior

    def report_constraint_counts(self):
        if not self.do_count:
            return
        print("Constraint counts per prior:")
        for i, count in zip(self.constraint_indices, self.constraint_counts):
            print("{}: {} ({:%})".format(self.names[i], count, count / self.total_tokens))
        print("All constraints: {} ({:%})".format(self.total_constraints, self.total_constraints / self.total_tokens))

    def describe(self):
        message = "\n".join(prior.describe() for prior in self.priors)
        return message

    def is_violated(self, actions, parent, sibling):
        for prior in self.priors:
            if isinstance(prior, Constraint):
                if prior.is_violated(actions, parent, sibling):
                    return True
        return False

    def at_once(self, actions, parent, sibling):
        """
        Given a full sequence of actions, parents, and siblings, each of shape
        (batch, time), *retrospectively* compute what was the joint prior at all
        time steps. The combined prior has shape (batch, time, L).
        """

        B, T = actions.shape
        zero_prior = np.zeros((B, T, self.L), dtype=np.float32) # (batch, time, L)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))] # i x (batch, time, L)

        # Set initial prior
        # Note: intial_prior() is already a combined prior, so we just set the
        # first individual prior, ind_priors[0].
        initial_prior = self.initial_prior() # Shape (L,)
        ind_priors[0][:, 0, :] = initial_prior # Broadcast to (batch, L)

        dangling = np.ones(B)
        for t in range(1, T): # For each time step
            # Update dangling based on previously sampled token
            dangling += self.library.arities[actions[:, (t - 1)]] - 1
            for i in range(len(self.priors)): # For each Prior
                # Compute the ith Prior at time step t
                prior = self.priors[i](actions[:, :t],
                                       parent[:, t],
                                       sibling[:, t],
                                       dangling) # Shape (batch, L)
                ind_priors[i][:, t, :] += prior

        # Combine all Priors
        combined_prior = sum(ind_priors) + zero_prior

        return combined_prior


class Prior():
    """Abstract class whose call method return logits."""

    def __init__(self, library):
        self.library = library
        self.L = library.L
        self.mask_val = -np.inf

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

        return "{}: No description available.".format(self.__class__.__name__)


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
            prior[mask, t] = self.mask_val
        return prior
    
    def is_violated(self, actions, parent, sibling):
        """
        Given a set of actions, tells us if a prior constraint has been violated 
        post hoc. 
        
        This is a generic version that will run using the __call__ function so that one
        does not have to write a function twice for both DSO and Deap. 
        
        >>>HOWEVER<<<
        
        Using this function is less optimal than writing a variant for Deap. So...
        
        If you create a constraint and find you use if often with Deap, you should gp ahead anf
        write the optimal version. 

        Returns
        -------
        violated : Bool
        """
        caller          = inspect.getframeinfo(inspect.stack()[1][0])
        
        warnings.warn("{} ({}) {} : Using a slower version of constraint for Deap. You should write your own.".format(caller.filename, caller.lineno, type(self).__name__))
        
        assert len(actions.shape) == 2, "Only takes in one action at a time since this is how Deap will use it."
        assert actions.shape[0] == 1, "Only takes in one action at a time since this is how Deap will use it."
        
        self.mask_val   = 1.0
        dangling        = np.ones((1), dtype=np.int32)
        
        # For each step in time, get the prior                                
        for t in range(actions.shape[1]):
            dangling    += self.library.arities[actions[:,t]] - 1   
            priors      = self.__call__(actions[:,:t], parent[:,t], sibling[:,t], dangling)
            
            # Does our action conflict with this prior?
            if priors[0,actions[0,t]]:
                return True
             
        return False
    
    def test_is_violated(self, actions, parent, sibling):
        r"""
            This allows one to call the generic version of "is_violated" for testing purposes
            from the derived classes even if they have an optimized version. 
        """
        return Constraint.is_violated(self, actions, parent, sibling)
    

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

        elif self.relationship == "lchild":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.logical_and(np.isin(parent, adj_parents), np.equal(sibling, self.L))
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "rchild":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.logical_and(np.isin(parent, adj_parents), np.less(sibling, self.L))
            prior = self.make_constraint(mask, self.targets)

        return prior

    def is_violated(self, actions, parent, sibling):

        if self.relationship == "descendant":
            violated = jit_check_constraint_violation_descendant_with_target_tokens(
                actions, self.targets, self.effectors, self.library.binary_tokens, self.library.unary_tokens)

        elif self.relationship == "child":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            violated = jit_check_constraint_violation(actions, self.targets, parent, adj_parents)

        elif self.relationship == "sibling":
            violated = jit_check_constraint_violation(actions, self.targets, sibling, self.effectors)
            if not violated:
                violated = jit_check_constraint_violation(actions, self.effectors, sibling, self.targets)

        elif self.relationship == "uchild":
            unary_effectors = np.intersect1d(self.effectors, self.library.unary_tokens)
            adj_unary_effectors = self.library.parent_adjust[unary_effectors]
            adj_effectors = self.library.parent_adjust[self.effectors]
            violated = jit_check_constraint_violation_uchild(actions, parent, sibling, self.targets, 
                                                     adj_unary_effectors, adj_effectors)

        return violated

    def validate(self):
        message = []
        if self.relationship in ["child", "descendant", "uchild", "lchild", "rchild"]:
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

    def describe(self):

        targets = ", ".join([self.library.names[t] for t in self.targets])
        effectors = ", ".join([self.library.names[t] for t in self.effectors])
        relationship = {
            "child" : "a child",
            "sibling" : "a sibling",
            "descendant" : "a descendant",
            "uchild" : "the only unique child",
            "lchild" : "the left child",
            "rchild" : "the right child"
        }[self.relationship]
        message = "{}: [{}] cannot be {} of [{}]." \
                  .format(self.__class__.__name__, targets, relationship, effectors)
        return message


class TrigConstraint(RelationalConstraint):
    """Class that constrains trig Tokens from being the descendants of trig
    Tokens."""

    def __init__(self, library):
        targets = library.trig_tokens
        effectors = library.trig_tokens
        super(TrigConstraint, self).__init__(library=library,
                                             targets=targets,
                                             effectors=effectors,
                                             relationship="descendant")
        
    def is_violated(self, actions, parent, sibling):
        
        # Call a slightly faster descendant computation since target is the same as effectors
        return jit_check_constraint_violation_descendant_no_target_tokens(\
                actions, self.effectors, self.library.binary_tokens, self.library.unary_tokens)


class ConstConstraint(RelationalConstraint):
    """Class that constrains the const Token from being the only unique child
    of all non-terminal Tokens."""

    def __init__(self, library):
        targets = library.const_token
        effectors = np.concatenate([library.unary_tokens,
                                    library.binary_tokens])

        super(ConstConstraint, self).__init__(library=library,
                                              targets=targets,
                                              effectors=effectors,
                                              relationship="uchild")


class NoInputsConstraint(Constraint):
    """Class that constrains sequences without input variables.

    NOTE: This *should* be a special case of RepeatConstraint, but is not yet
    supported."""

    def __init__(self, library):
        Prior.__init__(self, library)

    def validate(self):
        if len(self.library.float_tokens) == 0:
            message = "All terminal tokens are input variables, so all" \
                "sequences will have an input variable."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):
        # Constrain when:
        # 1) the expression would end if a terminal is chosen and
        # 2) there are no input variables
        mask = (dangling == 1) & \
               (np.sum(np.isin(actions, self.library.input_tokens), axis=1) == 0)
        prior = self.make_constraint(mask, self.library.float_tokens)
        return prior

    def is_violated(self, actions, parent, sibling):
        # Violated if no input tokens are found in actions
        tokens = self.library.input_tokens
        return bool(np.isin(tokens, actions).sum() == 0)

    def describe(self):
        message = "{}: Sequences contain at least one input variable Token.".format(self.__class__.__name__)
        return message


class InverseUnaryConstraint(Constraint):
    """Class that constrains each unary Token from being the child of its
    corresponding inverse unary Tokens."""

    def __init__(self, library):
        Prior.__init__(self, library)
        self.priors = []

        for target, effector in library.inverse_tokens.items():
            targets = [target]
            effectors = [effector]
            prior = RelationalConstraint(library=library,
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
        prior = sum([prior(actions, parent, sibling, dangling) for prior in self.priors])
        return prior

    def is_violated(self, actions, parent, sibling):

        for p in self.priors:
            if p.is_violated(actions, parent, sibling):
                return True

        return False

    def describe(self):
        message = [prior.describe() for prior in self.priors]
        return "\n{}: ".format(self.__class__.__name__).join(message)


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
            "{}: At least one of (min_, max_) must not be None.".format(self.__class__.__name__)
        self.min = min_
        self.max = max_
        self.n_objects = Program.n_objects
        self.tokens = library.actionize(tokens)

        assert min_ is None, "{}: Repeat minimum constraints are not yet " \
            "supported. This requires knowledge of length constraints.".format(self.__class__.__name__)

    def __call__(self, actions, parent, sibling, dangling):
        if self.n_objects > 1:
            _, i_batch = get_position(actions, self.library.arities, n_objects=self.n_objects)
            actions = np.copy(actions)
            mask = get_mask(i_batch, actions.shape[1])
            actions[mask == 0] = -1

        counts = np.sum(np.isin(actions, self.tokens), axis=1)
        prior = self.init_zeros(actions)
        if self.min is not None:
            raise NotImplementedError
        if self.max is not None:
            mask = counts >= self.max
            prior += self.make_constraint(mask, self.tokens)

        return prior

    def is_violated(self, actions, parent, sibling):
        return bool(np.isin(actions, self.tokens).sum() > self.max)

    def describe(self):
        names = ", ".join([self.library.names[t] for t in self.tokens])
        if self.min is None:
            message = "{}: [{}] cannot occur more than {} times."\
                .format(self.__class__.__name__, names, self.max)
        elif self.max is None:
            message = "{}: [{}] must occur at least {} times."\
                .format(self.__class__.__name__, names, self.min)
        else:
            message = "{}: [{}] must occur between {} and {} times."\
                .format(self.__class__.__name__, names, self.min, self.max)
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
        self.n_objects = Program.n_objects
        if self.n_objects > 1:
            assert self.max is not None, "Is max length constraint turned on? Max length constraint is required when n_objects > 1."

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

        if self.n_objects > 1:
            i, _ = get_position(actions, self.library.arities, n_objects=self.n_objects)

            if self.max is not None:
                remaining = self.max - (i + 1)
                mask = dangling >= remaining - 1 # constrain binary
                prior += self.make_constraint(mask, self.library.binary_tokens)
                mask = dangling == remaining # constrain unary
                prior += self.make_constraint(mask, self.library.unary_tokens)

            # Constrain terminals when dangling == 1 until selecting the
            # (min_length)th token
            if self.min is not None:
                mask = np.logical_and((i + 2) < self.min,
                                        dangling == 1)
                prior += self.make_constraint(mask, self.library.terminal_tokens)
        else:
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

    def is_violated(self, actions, parent, sibling):
        l = len(actions[0])
        if self.min is not None and l < self.min:
            return True
        if self.max is not None and l > self.max:
            return True

        return False

    def describe(self):
        message = []
        indent = " " * len(self.__class__.__name__) + "  "
        if self.min is not None:
            message.append("{}: Sequences have minimum length {}.".format(self.__class__.__name__, self.min))
        if self.max is not None:
            message.append(indent + "Sequences have maximum length {}.".format(self.max))
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

    def describe(self):
        """Describe the Prior."""

        return "{}: Activated.".format(self.__class__.__name__)


class SoftLengthPrior(Prior):
    """Class that puts a soft prior on length. Before loc, terminal probabilities
    are scaled by exp(-(t - loc) ** 2 / (2 * scale)) where dangling == 1. After
    loc, non-terminal probabilities are scaled by that number."""

    def __init__(self, library, loc, scale):

        Prior.__init__(self, library)

        self.loc = loc
        self.scale = scale
        self.n_objects = Program.n_objects

        self.terminal_mask = np.zeros((self.L,), dtype=np.bool)
        self.terminal_mask[self.library.terminal_tokens] = True

        self.nonterminal_mask = ~self.terminal_mask

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        t = actions.shape[1] # Current time

        if self.n_objects > 1:
            t_batch, _ = get_position(actions, self.library.arities, n_objects=self.n_objects)

            # Adjust the terminal or non-terminal logits
            logit_adjust = -(t_batch - self.loc) ** 2 / (2 * self.scale)

            # Find indices where we want to decrease p(terminal). Do this when dangling == 1 and t (for that action) < loc
            idxs = np.logical_and(t_batch < self.loc,
                                  dangling == 1)
            prior[idxs] += np.outer(logit_adjust[idxs], self.terminal_mask)

            # Find indices where we want to decrease p(non-terminal). Do this when t (for that action) > loc
            nonterm_idxs = (t_batch > self.loc)
            prior[nonterm_idxs] += np.outer(logit_adjust[nonterm_idxs], self.nonterminal_mask)

        else:
            # Adjustment to terminal or non-terminal logits
            logit_adjust = -(t - self.loc) ** 2 / (2 * self.scale)

            # Before loc, decrease p(terminal) where dangling == 1
            if t < self.loc:
                prior[dangling == 1] += self.terminal_mask * logit_adjust

            # After loc, decrease p(non-terminal)
            else:
                prior += self.nonterminal_mask * logit_adjust

        return prior

    def validate(self):
        if self.loc is None or self.scale is None:
            message = "'scale' and 'loc' arguments must be specified!"
            return message
        return None


class LanguageModelPrior(Prior):
    """Class that applies a prior based on a pre-trained language model."""

    def __init__(self, library, weight=1.0, **kwargs):

        Prior.__init__(self, library)

        self.lm = LM(library, **kwargs)
        self.weight = weight

    def initial_prior(self):

        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):

        """
        NOTE: This assumes that the prior is always called sequentially during
        sampling. This may break if calling the prior arbitrarily.
        """
        if actions.shape[1] == 1:
            self.lm.next_state = None

        action = actions[:, -1] # Current action
        prior = self.lm.get_lm_prior(action)
        prior *= self.weight

        return prior

    def validate(self):
        if self.weight is None:
            message = "Need to specify language model arguments."
            return message

