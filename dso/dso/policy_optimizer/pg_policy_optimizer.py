
import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy

class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer.

    Parameters
    ----------
    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    """
    def __init__(self,
            policy : Policy,
            debug : int = 0,
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        super()._setup_policy_optimizer(policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)

    def _set_loss(self):
        """Not used in TF2 eager execution - loss is computed in train_step."""
        pass

    def _preppend_to_summary(self):
        """Additional summary fields for PG."""
        pass

    @tf.function
    def _train_step_tf(self, actions, obs, priors, lengths, rewards, on_policy, baseline):
        """TF function for training step with gradient tape."""
        from dso.memory import Batch

        sampled_batch = Batch(
            actions=actions,
            obs=obs,
            priors=priors,
            lengths=lengths,
            rewards=rewards,
            on_policy=on_policy
        )

        with tf.GradientTape() as tape:
            neglogp, entropy = self.policy.make_neglogp_and_entropy(sampled_batch, self.entropy_gamma)

            # Entropy loss
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)

            # Policy gradient loss
            r = tf.cast(rewards, tf.float32)
            baseline_f32 = tf.cast(baseline, tf.float32)
            pg_loss = tf.reduce_mean((r - baseline_f32) * neglogp)

            # Total loss
            total_loss = entropy_loss + pg_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return total_loss, entropy_loss, pg_loss, gradients

    def train_step(self, baseline, sampled_batch, iteration=0):
        """Computes loss, trains model, and returns summaries."""

        total_loss, entropy_loss, pg_loss, gradients = self._train_step_tf(
            sampled_batch.actions,
            sampled_batch.obs,
            sampled_batch.priors,
            sampled_batch.lengths,
            sampled_batch.rewards,
            sampled_batch.on_policy,
            baseline
        )

        # Store grads and vars for summary
        self._last_grads_and_vars = list(zip(gradients, self.policy.trainable_variables))

        # Build summary dict
        loss_dict = {
            "total_loss": float(total_loss),
            "entropy_loss": float(entropy_loss),
            "pg_loss": float(pg_loss)
        }

        # Write summaries if enabled
        if self.summary and self.summary_writer is not None:
            self._write_summaries(iteration, loss_dict, sampled_batch, baseline)
            with self.summary_writer.as_default():
                tf.summary.scalar("pg_loss", pg_loss, step=iteration)

        return loss_dict
