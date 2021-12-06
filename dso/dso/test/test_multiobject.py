import pytest
import numpy as np
from copy import deepcopy

from dso.test.test_core import model
from dso.program import from_str_tokens
from dso.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
from dso.test.test_prior import assert_invalid, assert_valid, make_sequence, BATCH_SIZE
from dso.program import Program, from_tokens


@pytest.mark.parametrize("nobjs", [2, 3])
def test_multiobject_output(model, nobjs):

    # Update and setup model
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.setup()

    np.random.seed(0)
    X = np.random.random((100, 1))

    Program.set_n_objects(nobjs)

    if nobjs == 2:
        exec_tokens = "cos,x1,sin,x1"
        funcs = [np.cos, np.sin]
    elif nobjs == 3:
        exec_tokens = "exp,x1,cos,x1,sin,x1"
        funcs = [np.exp, np.cos, np.sin]
    p = from_str_tokens(exec_tokens) # build program
    np_out = np.array([funcs[j](X) for j in range(len(funcs))]).squeeze() # keep output from numpy functions
    prog_out = p.execute(X) # keep output from program execute
    np.testing.assert_array_almost_equal(np_out, prog_out) # assert outputs almost equal


def test_multiobject_repeat(model):
    Program.set_n_objects(2)
    config_prior_length = deepcopy(model.config_prior["length"])
    config_prior_length["min_"] = 3
    model.config_prior = {} # turn off all other priors
    model.config_prior["repeat"] = {
        "tokens" : ["sin", "cos"],
        "min_" : None, # Not yet supported
        "max_" : 2,
        "on": True
    }
    model.config_prior["length"] = config_prior_length
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    invalid_cases = []
    invalid_cases.append(["sin", "x1", "sin", "sin", "sin", "x1"])
    invalid_cases.append(["cos"] * 3 + ["x1"] + ["sin", "x1"])
    invalid_cases.append(["sin", "sin", "sin", "x1", "mul", "x1", "x1"])
    invalid_cases.append(["sin", "cos", "sin", "sin", "x1", "mul", "sin", "x1", "x1"])
    assert_invalid(model, invalid_cases, n_objects=Program.n_objects)

    valid_cases = []
    valid_cases.append(["sin", "sin", "x1", "sin", "sin", "x1"])
    valid_cases.append(["sin"] * 2 + ["x1"] + ["log"] * 2 + ["x1"])
    valid_cases.append(["log", "sin", "x1", "cos", "log", "x1"])
    valid_cases.append(["div", "x1", "cos", "cos", "x1"] + ["mul", "sin", "sin", "x1", "x1"])
    assert_valid(model, valid_cases, n_objects=Program.n_objects)


def test_multiobject_relational(model):
    Program.set_n_objects(2)

    # Constrain x1 - x1 or x1 / x1
    targets = "x1"
    effectors = "sub,div"

    # Need multiple input variables for this particular constraint otherwise this
    # RelationalConstraint cannot be used with the LengthConstraint simultaneously.
    model.config_task["dataset"] = "Nguyen-12"

    config_prior_length = deepcopy(model.config_prior["length"])
    config_prior_length["min_"] = None
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["relational"] = {
        "targets" : targets,
        "effectors" : effectors,
        "relationship" : "uchild",
        "on": True
    }
    model.config_prior["length"] = config_prior_length
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    # Test cases for n_objects = 2
    valid_cases = []
    valid_cases.append("mul,x1,x1,add,x1,x1")
    valid_cases.append("sub,x1,sin,x1,mul,x1,x1")
    assert_valid(model, valid_cases, n_objects=2)

    invalid_cases = []
    invalid_cases.append("sub,x1,x1,sin,x1")
    invalid_cases.append("div,x1,x1,cos,x1")
    assert_invalid(model, invalid_cases, n_objects=2)

    Program.set_n_objects(3)
    # Test cases for n_objects = 3
    valid_cases = []
    valid_cases.append("sin,x1,div,x1,sin,x1,mul,x1,x1")
    valid_cases.append("add,x1,x1,div,x1,sin,x1,sin,x1")
    valid_cases.append("div,x1,cos,x1,cos,x1,mul,x1,x1")
    assert_valid(model, valid_cases, n_objects=3)

    invalid_cases = []
    invalid_cases.append("sin,x1,div,x1,x1,sub,x1,x1")
    invalid_cases.append("sub,x1,x1,sin,x1,cos,x1")
    invalid_cases.append("mul,x1,x1,cos,x1,div,x1,sub,x1,x1")
    assert_valid(model, valid_cases, n_objects=3)


@pytest.mark.parametrize("minmaxnobj", [
    (10, 10, 2), (4, 30, 2), (None, 10, 2),
    (None, 10, 3),(4, 10, 3), (10, 10, 3)])
# NOTE: This test doesn't use a fixture cause n_objects has to be specified before building a fixture
def test_multiobject_length(model, minmaxnobj):
    """Test cases for LengthConstraint (for single- and multi-object Programs)."""

    min_, max_, n_objects = minmaxnobj
    Program.set_n_objects(n_objects)
    model.setup()

    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["length"] = {"min_" : min_, "max_" : max_, "on" : True}
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    # First, check that randomly generated samples do not violate constraints
    actions, _, _ = model.controller.sample(BATCH_SIZE)
    programs = [from_tokens(a) for a in actions]
    if n_objects == 1:
        lengths = [len(p.traversal) for p in programs]
    else:
        lengths = [len(trav) for p in programs for trav in p.traversals]
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


def test_multiobject_trig(model):

    Program.set_n_objects(2)
    config_prior_length = deepcopy(model.config_prior["length"])
    config_prior_length["min_"] = 2
    model.config_prior = {} # Turn off all other Priors
    model.config_prior["trig"] = {"on": True}
    model.config_prior["length"] = config_prior_length
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

    # Test cases for n_objects = 2
    valid_cases = []
    valid_cases.append("sin,x1,cos,x1")
    valid_cases.append("mul,sin,x1,cos,x1,cos,x1")
    valid_cases.append("sin,x1,add,cos,x1,sin,x1")
    assert_valid(model, valid_cases, n_objects=2)

    invalid_cases = []
    invalid_cases.append("sin,x1,sin,sin,x1")
    invalid_cases.append("sin,sub,x1,x1,sin,sin,x1")
    invalid_cases.append("sub,sub,sub,x1,x1,x1,sin,cos,x1")
    assert_invalid(model, invalid_cases, n_objects=2)

    # Test cases for n_objects = 3
    Program.set_n_objects(3)
    valid_cases = []
    valid_cases.append("sin,x1,sin,x1,cos,x1")
    valid_cases.append("mul,x1,x1,cos,x1,add,x1,x1")
    assert_valid(model, valid_cases, n_objects=3)

    invalid_cases = []
    invalid_cases.append("add,sub,x1,x1,sin,x1,div,x1,x1,sub,x1,x1")
    invalid_cases.append("sin,sub,x1,x1,sin,sin,x1,cos,cos,x1")
    assert_valid(model, valid_cases, n_objects=3)


# This is required when running sequential tests
Program.set_n_objects(1)
