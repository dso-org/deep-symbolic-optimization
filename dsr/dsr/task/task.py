"""Factory functions for generating symbolic search tasks."""

from dsr.task.regression.regression import make_regression_task
from dsr.task.control.control import make_control_task
from dsr.program import Program


def make_task(task_type, **config_task):
    """
    Factory function for reward function that maps a Progarm to a scalar.

    Parameters
    ----------

    task_type : str
        Type of task:
        "regression" : Symbolic regression task.
        "control" : Episodic reinforcement learning task.

    config_task : kwargs
        Task-specific arguments. See specifications of task_dict. Special key
        "name" is required, which defines the benchmark (i.e. dataset for
        regression; environment for control).

    Returns
    -------

    reward_function : function
        Reward function mapping program.Program object to scalar. Includes
        test argument for train vs test evaluation.

    eval_function : function
        Evaluation function mapping program.Program object to a dict of task-
        specific evaluation metrics (primitives). Special optional key "success"
        is used for determining early stopping during training.

    function_set : list
        List of allowable functions (see functions.py for supported functions).

    n_input_var : int
        Number of input variables.

    stochastic : bool
        Whether the reward function of the task is stochastic.
    """

    # Dictionary from task name to task factory function
    task_dict = {
        "regression" : make_regression_task,
        "control" : make_control_task
    }
    
    reward_function, eval_function, function_set, n_input_var, stochastic = task_dict[task_type](**config_task)
    return reward_function, eval_function, function_set, n_input_var, stochastic


def set_task(config_task):
    """Helper function to make set the Program class task and execute function
    from task config."""

    reward_function, eval_function, function_set, n_input_var, stochastic = make_task(**config_task)
    Program.set_reward_function(reward_function)
    Program.set_eval_function(eval_function)
    Program.set_library(function_set, n_input_var)
    Program.set_stochastic(stochastic)
