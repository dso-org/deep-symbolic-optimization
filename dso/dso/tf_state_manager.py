import tensorflow as tf
from dso.program import Program

from abc import ABC, abstractmethod


class StateManager(ABC):
    """
        This abstract class defines the methods that should be implemented for having custom state variables added to
        the RNN inputs in addition to parent, sibling, and action. ParentSiblingManager is a good starting point for
        new implementations
    """

    def setup_manager(self, controller):
        """
        Function called inside the controller to perform the needed initializations (e.g., if the tf context is needed)
        :param controller the controller class
        """
        self.controller = controller
        self.max_length = controller.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Performs needed convertions to process the custom state variables as tensors (for example, one-hot encoding)
        :param obs: observation to be converted
        :return: tensor version of observation
        """
        return

    @abstractmethod
    def get_initial_tensor_arrays(self):
        """
          builds initial tensorArrays for custom state variables
        :return: tuple with all TensorArrays
        """
        return

    @abstractmethod
    def get_obs_ph(self):
        """
        Returns placeholders to be used when building the dictionary
        :return: tuple of all placeholders
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
     Builds the correct state manager according to the parameter given in the config file.
    :param config: All the parameters belonging to the state manager
    :return: StateManager instance
    """
    manager_dict = {
        "parent_sibling": ParentSiblingManager
    }

    if config is None:
        config = {}

    #if no manager was specified, the backwards-compatible one is used
    manager_type = config.pop("type", "parent_sibling")

    manager_class = manager_dict[manager_type]
    #instantiates the desired class
    state_manager = manager_class(**config)

    return state_manager


class ParentSiblingManager(StateManager):
    """
    Class that only uses actions, parent, and sibling as observation
    """

    def __init__(self, observe_parent=True, observe_sibling=True, observe_action=False, embedding=False,
                 embedding_size=8):
        """
        :param observe_parent:  are parent observations used?
        :param observe_sibling: are sibling observations used?
        :param observe_action: are action observations used?
        :param embedding: Embed each observation?
        :param embedding_size: Size of embedding for each observation if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

        lib = Program.library
        n_choices = lib.L
        # Define input dimensions
        self.n_action_inputs = n_choices + 1  # lib tokens + empty token
        self.n_parent_inputs = n_choices + 1 - len(lib.terminal_tokens)  # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = n_choices + 1  # lib tokens + empty tokens

    def setup_manager(self, controller):
        super().setup_manager(controller)
        # Create embeddings if needed
        if self.embedding:
            with tf.variable_scope("embeddings",
                                   initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=0)):
                if self.observe_action:
                    self.action_embeddings = tf.get_variable("action_embeddings", [self.n_action_inputs, self.embedding_size],
                                                        trainable=True)
                if self.observe_parent:
                    self.parent_embeddings = tf.get_variable("parent_embeddings", [self.n_parent_inputs, self.embedding_size],
                                                        trainable=True)
                if self.observe_sibling:
                    self.sibling_embeddings = tf.get_variable("sibling_embeddings", [self.n_sibling_inputs, self.embedding_size],
                                                         trainable=True)

    def get_tensor_input(self, obs):
        action, parent, sibling = obs
        observations = []
        if self.observe_action:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.action_embeddings, action)
            else:
                x = tf.one_hot(action, depth=self.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.parent_embeddings, parent)
            else:
                x = tf.one_hot(parent, depth=self.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.sibling_embeddings, sibling)
            else:
                x = tf.one_hot(sibling, depth=self.n_sibling_inputs)
            observations.append(x)
        input_ = tf.concat(observations, -1)
        return input_

    def get_initial_tensor_arrays(self):
        action_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True) # Action inputs
        parent_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True) #parent input
        sibling_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True) # Sibling inputs
        return (action_tensor, parent_tensor, sibling_tensor)

    def get_obs_ph(self):
        obs_ph = (
            tf.placeholder(tf.int32, [None, self.max_length]),
            tf.placeholder(tf.int32, [None, self.max_length]),
            tf.placeholder(tf.int32, [None, self.max_length])
            )
        return obs_ph