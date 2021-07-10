import pytest

from dso.core import DeepSymbolicOptimizer

# We have two forms of doing the AbAg optimization: full and short
#   1) Full: generates the full sequence, including fixed positions. It has the
#      advantage of using the fixed positions as "context" for generating the mutable ones.
#   2) Short: generates only the positions that are allowed to mutate. It doesn't use
#      fixed positions as context. It's supposed to be faster (since we only sample a 
#      much smaller set of positions), but after running some tests, I noticed that the 
#      speed up is minor.

@pytest.mark.parametrize("config_file", ['test/binding_task/data/no_lm_prior/config_full.json',
                                         'test/binding_task/data/no_lm_prior/config_short.json'])
def test_task_execution(config_file):
    model = DeepSymbolicOptimizer(config_file)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.config_task['paths']['use_gpu'] = False
    model.train()
