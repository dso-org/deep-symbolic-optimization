import pytest
from pkg_resources import resource_filename

from dsr.core import DeepSymbolicOptimizer
from dsr.program import from_tokens, from_str_tokens, Program
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
import tensorflow as tf
import numpy as np

@pytest.fixture
def model():
    return DeepSymbolicOptimizer("config.json")

def test_model(model):
    # update and setup model
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.setup()

    exec_nums = np.arange(0, 5, step=0.5) # values to test through execute and numpy functions
    exec_tokens = ["cos,x1,sin,x1", "exp,x1,cos,x1,sin,x1"] # two different multi-object traversals to test
    np_fcns = [[lambda x: np.cos(x), lambda x: np.sin(x)], [lambda x: np.exp(x), lambda x: np.cos(x), lambda x: np.sin(x)]] # numpy versions of the multi-object traversals
    n_objs = [2, 3] # number of objects in each multi-object traversal
    for i, str_tokens in enumerate(exec_tokens):
        p = from_str_tokens(str_tokens, optimize=False, n_objects=n_objs[i]) # build program
        funcs = np_fcns[i] # grab matching set of numpy functions
        for n in exec_nums: # run execute and numpy functions over each test value
            np_out = np.array([funcs[j](n) for j in range(len(funcs))]).squeeze() # keep output from numpy functions
            prog_out = np.array(p.execute(np.array([[n]]))).squeeze() # keep output from program execute
            np.testing.assert_array_almost_equal(np_out, prog_out) # assert outputs almost equal
