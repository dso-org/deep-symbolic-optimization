import dso
from dso.library import Library, Token
from dso.functions import create_tokens
from dso.prior import Prior

"""
This file implements an "empty" task and prior for test purposes.
"""

def make_task(function_set, param):
    #Checking if the param wasn't suppressed for some reason.
    assert param == "test"
    function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]

    def reward(p):
        r = 0.
        return r

    def evaluate(p):
        info = {}
        return info

    tokens = create_tokens(n_input_var=1,
                           function_set=function_set,
                           protected=False,
                           decision_tree_threshold_set=None)

    library = Library(tokens)

    stochastic = False

    extra_info = {}

    task = dso.task.Task(reward_function=reward,
                         evaluate=evaluate,
                         library=library,
                         stochastic=stochastic,
                         task_type='test_task',
                         name="test_task",
                         extra_info=extra_info)

    return task


class CustomPrior(Prior):

    def __init__(self, library, param):
        Prior.__init__(self, library)
        #Just check whether the prior parameter got lost for some reason
        assert param == "test"

    def __call__(self, actions, parent, sibling, dangling):
        # Initialize the prior
        prior = self.init_zeros(actions)
        return prior

    def validate(self):
        return None