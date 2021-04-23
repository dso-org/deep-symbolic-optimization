"""Tests for various Priors."""

r"""
    Run from hypothesis_testing/dsr/dsr as:
    
        pytest test/test_prior.py
"""

import pytest

from dsr.core import DeepSymbolicOptimizer
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
from dsr.program import from_tokens, Program
from dsr.memory import Batch
from dsr.controller import parents_siblings
from dsr.subroutines import jit_parents_siblings_at_once
from dsr.prior import RepeatConstraint, RelationalConstraint, TrigConstraint, \
                      ConstConstraint, InverseUnaryConstraint, LengthConstraint

import numpy as np
import inspect


BATCH_SIZE = 1000


@pytest.fixture
def model():
    return DeepSymbolicOptimizer("config.json")


def make_failed_message(caller, i, n, msg):
    ">>> Test Failed! Caller: {} ({}) {} : TEST {}/{}: \"{}\" ".format(caller.filename, caller.lineno, caller.function, i+1, n, msg)


def make_testing_message(caller, i, n, msg):
    print(">>> Testing Caller: {} ({}) {} : TEST {}/{}: \"{}\" ".format(caller.filename, caller.lineno, caller.function, i+1, n, msg))
    

def assert_invalid(model, cases):
    cases = [Program.library.actionize(case) for case in cases]
    batch = make_batch(model, cases)
    logp = model.controller.compute_probs(batch, log=True)
    print(batch)
    assert all(np.isneginf(logp)), \
        "Found invalid case with probability > 0."


def assert_valid(model, cases):
    cases = [Program.library.actionize(case) for case in cases]
    batch = make_batch(model, cases)
    logp = model.controller.compute_probs(batch, log=True)
    assert all(logp > -np.inf), \
        "Found valid case with probability 0."


def pre_assert_is_violation(model, cases, prior_class, caller):
    assert callable(prior_class.is_violated)
    
    cases               = [Program.library.actionize(case) for case in cases]
    batch               = make_batch(model, cases)
    results             = []
    
    # For each action sequence in the batch.
    # Deap works one at a time, so we do it this way. 
    for i,a in enumerate(batch.actions):
        a  = np.expand_dims(a, axis=0)
        parents, siblings   = jit_parents_siblings_at_once(a,
                                                           arities=Program.library.arities, 
                                                           parent_adjust=Program.library.parent_adjust)
            
        r1 = prior_class.is_violated(a,parents,siblings)        # Tests an optimized version if we have one
        r2 = prior_class.test_is_violated(a,parents,siblings)   # Tests the slower universal version
    
        make_testing_message(caller, i, len(batch.actions), "{} == {}".format(r1,r2))
                        
        assert r1==r2, make_failed_message(caller, i, len(batch.actions), "Both methods should return the same results.")
    
        results.append(r1)
    
    return results
        

def assert_is_violation_true(model, cases, prior_class):
    caller  = inspect.getframeinfo(inspect.stack()[1][0])
    results = pre_assert_is_violation(model, cases, prior_class, caller)
    for i,r in enumerate(results):
        assert r, make_failed_message(caller, i, len(results), "Return value should be TRUE, but is not.")
 
    
def assert_is_violation_false(model, cases, prior_class):
    caller  = inspect.getframeinfo(inspect.stack()[1][0])
    results = pre_assert_is_violation(model, cases, prior_class, caller)
    for i,r in enumerate(results):
        assert not r, make_failed_message(caller, i, len(results), "Return value should be FALSE, but is not.")


def make_sequence(model, L):
    """Utility function to generate a sequence of length L"""
    X = Program.library.input_tokens[0]
    U = Program.library.unary_tokens[0]
    B = Program.library.binary_tokens[0]
    num_B = (L - 1) // 2
    num_U = int(L % 2 == 0)
    num_X = num_B + 1
    case = [B] * num_B + [U] * num_U + [X] * num_X
    assert len(case) == L
    case = case[:model.controller.max_length]
    return case


def make_batch(model, actions):
    """
    Utility function to generate a Batch from (unfinished) actions.

    This uses essentially the same logic as controller.py's loop_fn, except
    actions are prescribed instead of samples. Is there a way to refactor these
    with less code reuse?
    """

    batch_size = len(actions)
    L = model.controller.max_length

    # Pad actions to maximum length
    actions = np.array([np.pad(a, (0, L - len(a)), "constant")
                        for a in actions], dtype=np.int32)

    # Initialize obs
    prev_actions = np.zeros_like(actions)
    parents = np.zeros_like(actions)
    siblings = np.zeros_like(actions)

    arities = Program.library.arities
    parent_adjust = Program.library.parent_adjust

    # Set initial values
    empty_parent = np.max(parent_adjust) + 1
    empty_sibling = len(arities)
    action = empty_sibling
    parent, sibling = empty_parent, empty_sibling
    prior = np.array([model.prior.initial_prior()] * batch_size)

    priors = []
    lengths = np.zeros(batch_size, dtype=np.int32)
    finished = np.zeros(batch_size, dtype=np.bool_)
    dangling = np.ones(batch_size, dtype=np.int32)
    for i in range(L):
        partial_actions = actions[:, :(i + 1)]

        # Set prior and obs used to generate this action
        prev_actions[:, i] = action
        parents[:, i] = parent
        siblings[:, i] = sibling
        priors.append(prior)

        # Compute next obs and prior
        action = actions[:, i]
        parent, sibling = parents_siblings(tokens=partial_actions,
                                           arities=arities,
                                           parent_adjust=parent_adjust)
            
        dangling += arities[action] - 1
        prior = model.prior(partial_actions, parent, sibling, dangling)
        finished = np.where(np.logical_and(dangling == 0, lengths == 0),
                            True,
                            False)
        lengths = np.where(finished,
                           i + 1,
                           lengths)

    lengths = np.where(lengths == 0, L, lengths)
    obs = [prev_actions, parents, siblings]
    priors = np.array(priors).swapaxes(0, 1)
    rewards = np.zeros(batch_size, dtype=np.float32)
    on_policy = np.ones(batch_size, dtype=np.bool)
    batch = Batch(actions, obs, priors, lengths, rewards, on_policy)
    return batch


def test_repeat(model):
    """Test cases for RepeatConstraint."""

    model.config_prior = {} # Turn off all other Priors
    model.config_prior["repeat"] = {
        "tokens" : ["sin", "cos"],
        "min_" : None, # Not yet supported
        "max_" : 2
    }
    
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = RepeatConstraint(Program.library, **model.config_prior["repeat"])

    invalid_cases = []
    invalid_cases.append(["sin"] * 3)
    invalid_cases.append(["cos"] * 3)
    invalid_cases.append(["sin", "cos", "sin"])
    invalid_cases.append(["mul", "sin"] * 3)
    invalid_cases.append(["mul", "sin", "x1", "sin", "mul", "cos"])
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)

    valid_cases = []
    valid_cases.append(["mul"] + ["sin"] * 2 + ["log"] * 2)
    valid_cases.append(["sin"] + ["mul", "exp"] * 4 + ["cos"])
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)


def test_no_inputs(model):
    """Test cases for NoInputsConstraint."""

    # This test case needs a float Token before creating the model
    model.config["task"]["name"] = "Constant-1"
    model.pool = model.make_pool() # Resets Program.task with new Task

    model.config_prior = {} # Turn off all other Priors
    model.config_prior["no_inputs"] = {}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    invalid_cases = []
    invalid_cases.append("sin,const")
    invalid_cases.append("mul,const,const")
    invalid_cases.append("log,exp,const")
    assert_invalid(model, invalid_cases)

    valid_cases = []
    valid_cases.append(["sin"] * 10)
    valid_cases.append(["mul"] * 5)
    valid_cases.append("mul,const")
    valid_cases.append("mul,const,x1")
    valid_cases.append("mul,x1,x1")
    assert_valid(model, valid_cases)
    
    # No test for is_violation


def test_descendant(model):
    """Test cases for descendant RelationalConstraint."""

    descendants = "add,mul"
    ancestors = "exp,log"

    library = Program.library
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["relational"] = {
        "targets" : descendants,
        "effectors" : ancestors,
        "relationship" : "descendant"
    }

    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = RelationalConstraint(Program.library, **model.config_prior["relational"])

    descendants = library.actionize(descendants)
    ancestors = library.actionize(ancestors)

    U = [i for i in library.unary_tokens
         if i not in ancestors and i not in descendants][0]
    B = [i for i in library.binary_tokens
         if i not in ancestors and i not in descendants][0]

    # For each D-A combination, generate invalid cases where A is an ancestor
    # of D
    invalid_cases = []
    for A in ancestors:
        for D in descendants:
            invalid_cases.append([A, D])
            invalid_cases.append([A] * 10 + [D])
            invalid_cases.append([A] + [U, B] * 5 + [D])
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)

    # For each D-A combination, generate valid cases where A is not an ancestor
    # of D
    valid_cases = []
    for A in ancestors:
        for D in descendants:
            valid_cases.append([U, D])
            valid_cases.append([D] + [U] * 10 + [A])
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)


def test_trig(model):
    """Test cases for TrigConstraint."""

    library = Program.library
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["trig"] = {}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = TrigConstraint(Program.library, **model.config_prior["trig"])

    X = library.input_tokens[0]
    U = [i for i in library.unary_tokens
         if i not in library.trig_tokens][0]
    B = library.binary_tokens[0]

    # For each trig-trig combination, generate invalid cases where one Token is
    # a descendant the other
    invalid_cases = []
    trig_tokens = library.trig_tokens
    for t1 in trig_tokens:
        for t2 in trig_tokens:
            invalid_cases.append([t1, t2, X]) # E.g. sin(cos(x))
            invalid_cases.append([t1, B, X, t2, X]) # E.g. sin(x + cos(x))
            invalid_cases.append([t1] + [U] * 10 + [t2, X])
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)

    # For each trig-trig pair, generate valid cases where one Token is the
    # sibling the other
    valid_cases = []
    for t1 in trig_tokens:
        for t2 in trig_tokens:
            valid_cases.append([B, U, t1, X, t2, X]) # E.g. log(sin(x)) + cos(x)
            valid_cases.append([B, t1, X, t2, X]) # E.g. sin(x) + cos(x)
            valid_cases.append([U] + valid_cases[-1]) # E.g. log(sin(x) + cos(x))
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)


def test_child(model):
    """Test cases for child RelationalConstraint."""

    library = Program.library
    parents = library.actionize("log,exp,mul")
    children = library.actionize("exp,log,sin")

    print(parents)
    print(children)

    model.config_prior = {} # Turn off all other Priors
    model.config_prior["relational"] = {
        "targets" : children,
        "effectors" : parents,
        "relationship" : "child"
    }
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    prior_class = RelationalConstraint(Program.library, **model.config_prior["relational"])

    # For each parent-child pair, generate invalid cases where child is one of
    # parent's children.
    X = library.input_tokens[0]
    assert X not in children, \
        "Error in test case specification. Do not include x1 in children."
    invalid_cases = []
    for p, c in zip(parents, children):
        arity = library.tokenize(p)[0].arity
        for i in range(arity):
            before = i
            after = arity - i - 1
            case = [p] + [X] * before + [c] + [X] * after
            invalid_cases.append(case)
    print(invalid_cases)
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)


def test_uchild(model):
    """Test cases for uchild RelationalConstraint."""

    library = Program.library
    targets = library.actionize("x1")
    effectors = library.actionize("sub,div") # i.e. no x1 - x1 or x1 / x1

    model.config_prior = {} # Turn off all other Priors
    model.config_prior["relational"] = {
        "targets" : targets,
        "effectors" : effectors,
        "relationship" : "uchild"
    }
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = RelationalConstraint(Program.library, **model.config_prior["relational"])

    # Generate valid test cases
    valid_cases = []
    valid_cases.append("mul,x1,x1")
    valid_cases.append("sub,x1,sub,x1,sub,x1,sin,x1")
    valid_cases.append("sub,sub,sub,x1,sin,x1,x1")
    valid_cases.append("sub,sin,x1,sin,x1")
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)

    # Generate invalid test cases
    invalid_cases = []
    invalid_cases.append("add,sub,x1,x1,sin,x1")
    invalid_cases.append("sin,sub,x1,x1")
    invalid_cases.append("sub,sub,sub,x1,x1,x1")
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)


def test_const(model):
    """Test cases for ConstConstraint."""

    # This test case needs the const Token before creating the model
    model.config["task"]["name"] = "Constant-1"
    model.pool = model.make_pool() # Resets Program.task with new Task

    library = Program.library
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["const"] = {}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    prior_class = ConstConstraint(Program.library, **model.config_prior["const"])

    # Generate valid test cases
    valid_cases = []
    valid_cases.append("mul,const,x1")
    valid_cases.append("sub,const,sub,const,x1")
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)

    # Generate invalid test cases
    invalid_cases = []
    invalid_cases.append("sin,const")
    invalid_cases.append("mul,const,const")
    invalid_cases.append("sin,add,const,const")
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)


def test_sibling(model):
    """Test cases for sibling RelationalConstraint."""

    library = Program.library
    targets = library.actionize("sin,cos")
    effectors = library.actionize("x1")

    model.config_prior = {} # Turn off all other Priors
    model.config_prior["relational"] = {
        "targets" : targets,
        "effectors" : effectors,
        "relationship" : "sibling"
    }
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = RelationalConstraint(Program.library, **model.config_prior["relational"])

    # Generate valid test cases
    valid_cases = []
    valid_cases.append("mul,sin,x1,cos,x1")
    valid_cases.append("sin,cos,x1")
    valid_cases.append("add,add,sin,mul,x1,x1,cos,x1,x1")
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)

    # Generate invalid test cases
    invalid_cases = []
    invalid_cases.append("add,x1,sin,x1")
    invalid_cases.append("add,sin,x1,x1")
    invalid_cases.append("add,add,sin,mul,x1,x1,x1,sin,x1")
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)


def test_inverse(model):
    """Test cases for InverseConstraint."""

    library = Program.library
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["inverse"] = {}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    prior_class = InverseUnaryConstraint(Program.library, **model.config_prior["inverse"])

    # Generate valid cases
    valid_cases = []
    valid_cases.append("exp,sin,log,cos,exp,x1")
    valid_cases.append("mul,sin,log,x1,exp,cos,x1")
    assert_valid(model, valid_cases)
    assert_is_violation_false(model, valid_cases, prior_class)

    # Generate invalid cases for each inverse
    invalid_cases = []
    invalid_cases.append("mul,sin,x1,exp,log,x1")
    for t1, t2 in library.inverse_tokens.items():
        invalid_cases.append([t1, t2])
        invalid_cases.append([t2, t1])
    assert_invalid(model, invalid_cases)
    assert_is_violation_true(model, invalid_cases, prior_class)


@pytest.mark.parametrize("minmax", [(10, 10), (4, 30), (None, 10), (10, None)])
def test_length(model, minmax):
    """Test cases for LengthConstraint."""

    min_, max_ = minmax
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["length"] = {"min_" : min_, "max_" : max_}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()
    
    # First, check that randomly generated samples do not violate constraints
    actions, _, _ = model.controller.sample(BATCH_SIZE)
    programs = [from_tokens(a, optimize=True) for a in actions]
    lengths = [len(p.traversal) for p in programs]
    if min_ is not None:
        min_L = min(lengths)
        assert min_L >= min_, \
            "Found min length {} but constrained to {}.".format(min_L, min_)
    if max_ is not None:
        max_L = max(lengths)
        assert max_L <= max_, \
            "Found max length {} but constrained to {}.".format(max_L, max_)

    # Next, check valid and invalid test cases based on min_ and max_
    # Valid test cases should not be constrained
    # Invalid test cases should all be constrained
    valid_cases = []
    invalid_cases = []

    # Initial prior prevents length-1 tokens
    case = make_sequence(model, 1)
    invalid_cases.append(case)

    if min_ is not None:
        # Generate an invalid case that is one Token too short
        if min_ > 1:
            case = make_sequence(model, min_ - 1)
            invalid_cases.append(case)

        # Generate a valid case that is exactly the minimum length
        case = make_sequence(model, min_)
        valid_cases.append(case)

    if max_ is not None:
        # Generate an invalid case that is one Token too long (which will be
        # truncated to dangling == 1)
        case = make_sequence(model, max_ + 1)
        invalid_cases.append(case)

        # Generate a valid case that is exactly the maximum length
        case = make_sequence(model, max_)
        valid_cases.append(case)

    assert_valid(model, valid_cases)
    assert_invalid(model, invalid_cases)

    
