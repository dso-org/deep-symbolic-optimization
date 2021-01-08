import pytest
from pkg_resources import resource_filename

from dsr.core import DeepSymbolicOptimizer
from dsr.program import from_tokens, from_str_tokens, Program
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
import tensorflow as tf
import numpy as np

@pytest.fixture
def model():
    return DeepSymbolicOptimizer("../config_test_multiobject.json")

def test_model(model):
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.setup()
    exec_nums = np.arange(0, 5, step=0.5)
    exec_tokens = ["cos,x1,sin,x1", "exp,x1,cos,x1,sin,x1"]
    np_fcns = [[lambda x: np.cos(x), lambda x: np.sin(x)], [lambda x: np.exp(x), lambda x: np.cos(x), lambda x: np.sin(x)]]
    n_objs = [2, 3]
    for i, str_tokens in enumerate(exec_tokens):
        p = from_str_tokens(str_tokens, optimize=False, n_objects=n_objs[i])
        funcs = np_fcns[i]
        for n in exec_nums:
            np_out = np.array([funcs[j](n) for j in range(len(funcs))]).squeeze()
            prog_out = np.array(p.execute(np.array([[n]]))).squeeze()
            np.testing.assert_array_almost_equal(np_out, prog_out)
