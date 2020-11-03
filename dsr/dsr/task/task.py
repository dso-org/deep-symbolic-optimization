"""Factory functions for generating symbolic search tasks."""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any

from dsr.task.regression.regression import make_regression_task
from dsr.task.control.control import make_control_task
from dsr.program import Program


@dataclass(frozen=True)
class Task:
    """
    Data object specifying a symbolic search task.

    Attributes
    ----------
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

    extra_info : dict
        Extra task-specific info, e.g. reference to symbolic policies for
        control task.
    """

    reward_function: Callable[[Program], float]
    evaluate: Callable[[Program], float]
    function_set: List[Any]
    n_input_var: int
    stochastic: bool
    extra_info: Dict[str, Any]


def make_task(task_type, **config_task):
    """
    Factory function for Task object.

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

    task : Task
        Task object.
    """

    # Dictionary from task name to task factory function
    task_dict = {
        "regression" : make_regression_task,
        "control" : make_control_task
    }
    
    task = task_dict[task_type](**config_task)
    return task


def set_task(config_task):
    """Helper function to make set the Program class Task, execute function,
    and library from task config."""

    # Use of protected functions is the same for all tasks, so it's handled separately
    protected = config_task.pop("protected") if "protected" in config_task else True

    Program.set_execute(protected)
    task = make_task(**config_task)
    Program.set_task(task)
    Program.set_library(task.function_set, task.n_input_var, protected)
