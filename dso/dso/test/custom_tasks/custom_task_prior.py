"""
This file implements an "empty" task and prior for test purposes.
"""

from dso.task import HierarchicalTask
from dso.library import Library
from dso.functions import create_tokens
from dso.prior import Prior


class CustomTask(HierarchicalTask):

    task_type = "dummy"

    def __init__(self, param):

        super(HierarchicalTask).__init__()

        self.param = param

        # Create a Library
        tokens = create_tokens(n_input_var=1,
                               function_set=["add", "sin"],
                               protected=False,
                               decision_tree_threshold_set=None)
        self.library = Library(tokens)

        self.stochastic = False
        self.name = "dummy"

    def reward_function(self, p):
        r = 0.
        return r

    def evaluate(self, p):
        info = {}
        return info


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