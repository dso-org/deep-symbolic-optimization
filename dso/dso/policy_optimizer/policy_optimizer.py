from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.memory import Batch
from dso.utils import import_custom_source
from dso.policy.policy import Policy

# Used for function annotations using the type system
summaries = dict

def make_policy_optimizer(policy, policy_optimizer_type, **config_policy_optimizer):
    """Factory function for policy optimizer object."""

    if policy_optimizer_type == "pg":
        from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer
        policy_optimizer_class = PGPolicyOptimizer
    elif policy_optimizer_type == "pqt":
        from dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer
        policy_optimizer_class = PQTPolicyOptimizer
    elif policy_optimizer_type == "ppo":
        from dso.policy_optimizer.ppo_policy_optimizer import PPOPolicyOptimizer
        policy_optimizer_class = PPOPolicyOptimizer
    else:
        # Custom policy import
        policy_optimizer_class = import_custom_source(policy_optimizer_type)
        assert issubclass(policy_optimizer_class, PolicyOptimizer), \
                "Custom policy optimizer {} must subclass dso.policy_optimizer.PolicyOptimizer.".format(policy_optimizer_class)

    policy_optimizer = policy_optimizer_class(policy,
                                              **config_policy_optimizer)

    return policy_optimizer

class PolicyOptimizer(ABC):
    """Abstract class for a policy optimizer. A policy optimizer is an
    algorithm for optimizing the parameters of a parametrized policy.

    To define a new optimizer, inherit from this class and add the following
    methods (look in _setup_policy_optimizer below):

        _set_loss() : Define the \propto \log(p(\tau|\theta)) loss for the method
        _preppend_to_summary() : Add additional fields for the tensorflow summary

    """

    def _init(self,
            policy : Policy,
            debug : int = 0,
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        '''Parameters
        ----------
        policy : dso.policy.Policy
            Parametrized probability distribution over discrete objects

        debug : int
            Debug level, also used in learn(). 0: No debug. 1: Print shapes and
            number of parameters for each variable.

        summary : bool
            Write tensorboard summaries?

        optimizer : str
            Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

        learning_rate : float
            Learning rate for optimizer.

        entropy_weight : float
            Coefficient for entropy bonus.

        entropy_gamma : float or None
            Gamma in entropy decay. None (or
            equivalently, 1.0) turns off entropy decay.
        '''
        self.policy = policy

        # Needed in _setup_optimizer
        self.debug = debug
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

        # Need in self.summary
        self.summary = summary
        self.summary_writer = None

        # Needed for make_batch_spec calls
        self.n_choices = Program.library.L

        # Need in _init_loss_with_entropy
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma

        # Create optimizer (TF2 Keras style)
        self.optimizer = self._make_optimizer(optimizer, learning_rate)

    def _make_optimizer(self, name, learning_rate):
        """Create a Keras optimizer."""
        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.99)
        if name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        raise ValueError("Did not recognize optimizer '{}'".format(name))

    def _compute_loss(self, sampled_batch, baseline):
        """Compute loss including entropy term. To be overridden by subclasses."""
        neglogp, entropy = self.policy.make_neglogp_and_entropy(sampled_batch, self.entropy_gamma)

        # Entropy loss
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)

        # Base loss (to be extended by subclass)
        return neglogp, entropy, entropy_loss

    @abstractmethod
    def _set_loss(self) -> None:
        """Define the \propto \log(p(\tau|\theta)) loss for the method.
        This is called during training to compute the full loss.

        Returns
        -------
            None
        """
        raise NotImplementedError

    def _setup_optimizer(self):
        """Setup debug printing if needed."""
        if self.debug >= 1:
            total_parameters = 0
            print("")
            for variable in self.policy.trainable_variables:
                shape = variable.shape
                n_parameters = np.prod(shape)
                total_parameters += n_parameters
                print("Variable:    ", variable.name)
                print("  Shape:     ", shape)
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)

    # abstractmethod (override if needed)
    def _preppend_to_summary(self) -> None:
        """Add particular fields to the summary log.
        Override if needed.
        """
        pass

    def setup_summary_writer(self, summary_dir):
        """Setup TensorFlow summary writer for TensorBoard."""
        if self.summary:
            self.summary_writer = tf.summary.create_file_writer(summary_dir)

    def _write_summaries(self, iteration, loss_dict, sampled_batch, baseline):
        """Write summaries to TensorBoard."""
        if self.summary_writer is None:
            return {}

        with self.summary_writer.as_default():
            tf.summary.scalar("entropy_loss", loss_dict.get("entropy_loss", 0), step=iteration)
            tf.summary.scalar("total_loss", loss_dict.get("total_loss", 0), step=iteration)
            tf.summary.scalar("reward", tf.reduce_mean(sampled_batch.rewards), step=iteration)
            tf.summary.scalar("baseline", baseline, step=iteration)
            tf.summary.histogram("reward", sampled_batch.rewards, step=iteration)
            tf.summary.histogram("length", sampled_batch.lengths, step=iteration)

            # Log gradients and variables
            if hasattr(self, '_last_grads_and_vars'):
                for g, v in self._last_grads_and_vars:
                    if g is not None:
                        tf.summary.histogram(v.name, v, step=iteration)
                        tf.summary.scalar(v.name + '_norm', tf.norm(v), step=iteration)
                        tf.summary.histogram(v.name + '_grad', g, step=iteration)
                        tf.summary.scalar(v.name + '_grad_norm', tf.norm(g), step=iteration)

                grads = [g for g, v in self._last_grads_and_vars if g is not None]
                if grads:
                    tf.summary.scalar('gradient_norm', tf.linalg.global_norm(grads), step=iteration)

        return loss_dict


    def _setup_policy_optimizer(self,
            policy : Policy,
            debug : int = 0,
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        """Setup of the policy optimizer.
        """
        self._init(policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)
        self._setup_optimizer()


    @abstractmethod
    def train_step(self,
            baseline : np.ndarray,
            sampled_batch : Batch,
            iteration : int = 0) -> summaries:
        """Computes loss, trains model, and returns summaries.

        Returns
        -------
            None
        """
        raise NotImplementedError


