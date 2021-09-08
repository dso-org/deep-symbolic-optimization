"""Factory functions for generating symbolic search tasks."""

from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import numpy as np

from dso.program import Program
from dso.library import Library
from dso.utils import import_custom_source
from dso.subroutines import parents_siblings


# The task class was set to frozen=False so that it can hold data necessary for computing the transition function
@dataclass(frozen=False)
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

    get_next_obs : function
        Function to process the state transition and return the next observations.
        Receives as input the last observations and the actions applied, and return
        a tuple with obs, prior, dangling

    reset_task: function
        Function that signals that the task will start, and returns the initial state.
        The input is <state_manager, prior>, and
        the initial observations are returned.

    library : Library
        Library of Tokens.

    stochastic : bool
        Whether the reward function of the task is stochastic.

    task_type : str
        Task type: regression or control.

    name : str
        Unique name for instance of this task.

    extra_info : dict
        Extra task-specific info, e.g. reference to symbolic policies for
        control task.
    """

    reward_function: Callable[[Program], float]
    evaluate: Callable[[Program], float]
    get_next_obs: Callable[[tuple, np.array], tuple]
    reset_task: Callable[[object, object], tuple]
    library: Library
    stochastic: bool
    task_type: str
    name: str
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
        Task-specific arguments. See specifications of task_dict.

    Returns
    -------

    task : Task
        Task object.
    """

    # Custom task import
    if task_type not in ['binding', 'regression', 'control']:
        make_function = import_custom_source(task_type)

    # Lazy import of task factory functions
    if task_type == 'binding':
        from dso.task.binding.binding import make_binding_task
        make_function = make_binding_task
    elif task_type == "regression":
        from dso.task.regression.regression import make_regression_task
        make_function = make_regression_task
    elif task_type == "control":
        from dso.task.control.control import make_control_task
        make_function = make_control_task

    task = make_function(**config_task)
    return task


def set_task(config_task):
    """Helper function to make set the Program class Task and execute function
    from task config."""

    # Use of protected functions is the same for all tasks, so it's handled separately
    protected = config_task["protected"] if "protected" in config_task else False

    Program.set_execute(protected)
    task = make_task(**config_task)
    Program.set_task(task)


"""
    All the functions below this point are default "helper" functions to help building tasks that use parent, sibling, 
    action as state features
"""


# Given the actions chosen so far, return the observation, the prior, and the updated dangling
def get_next_obs_parent_sibling(actions, dangling):
    self = Program.task # Gets a reference to the task, since it is not possible to recover "self" from pyfunc
    lib = Program.library
    n = actions.shape[0]  # Batch size
    i = actions.shape[1] - 1  # Current index
    action = actions[:, -1]  # Current action

    # Depending on the constraints, may need to compute parents and siblings
    if self.compute_parents_siblings:
        parent, sibling = parents_siblings(actions, arities=lib.arities, parent_adjust=lib.parent_adjust)
    else:
        parent = np.zeros(n, dtype=np.int32)
        sibling = np.zeros(n, dtype=np.int32)

    # Update dangling with (arity - 1) for each element in action
    dangling += lib.arities[action] - 1

    prior = self.prior(actions, parent, sibling, dangling)

    # Stacked because py_func does not allow returning a tuple.
    obs = np.stack([action, parent, sibling])
    return obs, prior, dangling


def reset_task_parent_sibling(state_manager, prior):
    """
    Returns initial observation (all empty tokens)
    :return:
    """
    self = Program.task
    self.prior = prior
    self.observe_parent = state_manager.observe_parent
    self.observe_sibling = state_manager.observe_sibling
    lib = Program.library
    n_choices = lib.L
    # Define input dimensions
    n_action_inputs = state_manager.n_action_inputs  # lib tokens + empty token
    n_parent_inputs = state_manager.n_parent_inputs  # Parent sub-lib tokens + empty token
    n_sibling_inputs = state_manager.n_sibling_inputs  # lib tokens + empty tokens

    self.compute_parents_siblings = any([self.observe_parent,
                                         self.observe_sibling,
                                         self.prior.requires_parents_siblings])

    initial_obs = []
    for n in [n_action_inputs, n_parent_inputs, n_sibling_inputs]:
        obs = np.array(n - 1, dtype=np.int32)
        initial_obs.append(obs)
    return initial_obs