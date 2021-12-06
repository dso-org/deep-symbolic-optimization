"""Factory functions for generating symbolic search tasks."""

from abc import ABC, abstractmethod
import numpy as np

from dso.program import Program
from dso.utils import import_custom_source
from dso.subroutines import parents_siblings


class Task(ABC):
    """
    Object specifying a symbolic search task.

    Attributes
    ----------
    library : Library
        Library of Tokens.

    stochastic : bool
        Whether the reward function of the task is stochastic.

    task_type : str
        Task type: regression or control

    name : str
        Unique name for instance of this task.
    """

    task_type = None

    @abstractmethod
    def reward_function(self, program):
        """
        The reward function for this task.

        Parameters
        ----------
        program : dso.program.Program

            The Program to compute reward of.

        Returns
        -------
        reward : float

            Fitness/reward of the program.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, program):
        """
        The evaluation metric for this task.

        Parameters
        ----------
        program : dso.program.Program

            The Program to evaluate.

        Returns
        -------

        info : dict

            Dictionary of evaluation metrics. Special key "success" is used to
            trigger early stopping.
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_obs(self, actions, obs):
        """
        Produce the next observation and prior from the current observation and
        list of actions so far. Observations must be 1-D np.float32 vectors.

        Parameters
        ----------

        actions : np.ndarray (dtype=np.int32)
            Actions selected so far, shape (batch_size, current_length)

        obs : np.ndarray (dtype=np.float32)
            Previous observation, shape (batch_size, OBS_DIM).

        Returns
        -------

        next_obs : np.ndarray (dtype=np.float32)
            The next observation, shape (batch_size, OBS_DIM).

        prior : np.ndarray (dtype=np.float32)
            Prior for selecting the next token, shape (batch_size,
            self.library.L).
        """
        pass

    @abstractmethod
    def reset_task(self):
        """
        Create the starting observation.

        Returns
        -------

        obs : np.ndarray (dtype=np.float32)
            Starting observation, shape (batch_size, OBS_DIM).
        """
        pass


class HierarchicalTask(Task):
    """
    A Task in which the search space is a binary tree. Observations include
    the previous action, the parent, the sibling, and/or the number of dangling
    (unselected) nodes.
    """

    OBS_DIM = 4 # action, parent, sibling, dangling

    def __init__(self):
        super(Task).__init__()

    def get_next_obs(self, actions, obs):

        dangling = obs[:, 3] # Shape of obs: (?, 4)
        action = actions[:, -1] # Current action
        lib = self.library

        # Compute parents and siblings
        parent, sibling = parents_siblings(actions,
                                           arities=lib.arities,
                                           parent_adjust=lib.parent_adjust,
                                           empty_parent=lib.EMPTY_PARENT,
                                           empty_sibling=lib.EMPTY_SIBLING)

        # Update dangling with (arity - 1) for each element in action
        dangling += lib.arities[action] - 1

        prior = self.prior(actions, parent, sibling, dangling) # (?, n_choices)

        # Reset initial values when tree completes
        if Program.n_objects > 1: # NOTE: do this to save computuational cost only when n_objects > 1
            finished = (dangling == 0)
            dangling[finished] = 1
            action[finished] = lib.EMPTY_ACTION
            parent[finished] = lib.EMPTY_PARENT
            sibling[finished] = lib.EMPTY_SIBLING
            prior[finished] = self.prior.initial_prior()

        next_obs = np.stack([action, parent, sibling, dangling], axis=1) # (?, 4)
        next_obs = next_obs.astype(np.float32)
        return next_obs, prior

    def reset_task(self, prior):
        """
        Returns the initial observation: empty action, parent, and sibling, and
        dangling is 1.
        """

        self.prior = prior

        # Order of observations: action, parent, sibling, dangling
        initial_obs = np.array([self.library.EMPTY_ACTION,
                                self.library.EMPTY_PARENT,
                                self.library.EMPTY_SIBLING,
                                1],
                               dtype=np.float32)
        return initial_obs


class SequentialTask(Task):
    """
    A Task in which the search space is a (possibly variable-length) sequence.
    The observation is simply the previous action.
    """

    pass


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

    # Lazy import of task factory functions

    if task_type == "regression":
        from dso.task.regression.regression import RegressionTask
        task_class = RegressionTask
    elif task_type == "control":
        from dso.task.control.control import ControlTask
        task_class = ControlTask
    else:
        # Custom task import
        task_class = import_custom_source(task_type)
        assert issubclass(task_class, Task), \
            "Custom task {} must subclass dso.task.Task.".format(task_class)

    task = task_class(**config_task)
    return task


def set_task(config_task):
    """Helper function to make set the Program class Task and execute function
    from task config."""

    # Use of protected functions is the same for all tasks, so it's handled separately
    protected = config_task["protected"] if "protected" in config_task else False

    Program.set_execute(protected)
    task = make_task(**config_task)
    Program.set_task(task)
