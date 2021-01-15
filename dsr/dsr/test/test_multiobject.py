from dsr.program import from_str_tokens
from dsr.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
from dsr.test.test_core import model
import numpy as np

def test_multiobject_output(model):
    # update and setup model
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.setup()

    np.random.seed(0)
    X = np.random.random((100,1))

    exec_tokens = ["cos,x1,sin,x1", "exp,x1,cos,x1,sin,x1"] # two different multi-object traversals to test
    np_fcns = [[np.cos, np.sin], [np.exp, np.cos, np.sin]]
    n_objs = [2, 3] # number of objects in each multi-object traversal
    for i, str_tokens in enumerate(exec_tokens):
        p = from_str_tokens(str_tokens, optimize=False, n_objects=n_objs[i]) # build program
        funcs = np_fcns[i] # grab matching set of numpy functions
        np_out = np.array([funcs[j](X) for j in range(len(funcs))]).squeeze() # keep output from numpy functions
        prog_out = np.array(p.execute(X)).squeeze() # keep output from program execute
        np.testing.assert_array_almost_equal(np_out, prog_out) # assert outputs almost equal
