
import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy

class PQTPolicyOptimizer(PolicyOptimizer):
    """Priority Queue Training policy gradient policy optimizer.

    Parameters
    ----------
     pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

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
            entropy_gamma : float = 1.0,
            # PQT hyperparameters
            pqt_k : int = 10,
            pqt_batch_size : int = 1,
            pqt_weight : float = 200.0,
            pqt_use_pg: bool = False) -> None:
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg
        super()._setup_policy_optimizer(policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)

    def _set_loss(self):
        """Not used in TF2 eager execution - loss is computed in train_step."""
        pass

    def _preppend_to_summary(self):
        """Additional summary fields for PQT."""
        pass

    @tf.function
    def _train_step_tf(self, actions, obs, priors, lengths, rewards, on_policy,
                       pqt_actions, pqt_obs, pqt_priors, pqt_lengths, pqt_rewards, pqt_on_policy,
                       baseline):
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

        pqt_batch = Batch(
            actions=pqt_actions,
            obs=pqt_obs,
            priors=pqt_priors,
            lengths=pqt_lengths,
            rewards=pqt_rewards,
            on_policy=pqt_on_policy
        )

        with tf.GradientTape() as tape:
            neglogp, entropy = self.policy.make_neglogp_and_entropy(sampled_batch, self.entropy_gamma)

            # Entropy loss
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)

            # PQT loss
            pqt_neglogp, _ = self.policy.make_neglogp_and_entropy(pqt_batch, self.entropy_gamma)
            pqt_loss = self.pqt_weight * tf.reduce_mean(pqt_neglogp)

            # Total loss
            total_loss = entropy_loss + pqt_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return total_loss, entropy_loss, pqt_loss, gradients

    def train_step(self, baseline, sampled_batch, pqt_batch, iteration=0):
        """Computes loss, trains model, and returns summaries."""

        total_loss, entropy_loss, pqt_loss, gradients = self._train_step_tf(
            sampled_batch.actions,
            sampled_batch.obs,
            sampled_batch.priors,
            sampled_batch.lengths,
            sampled_batch.rewards,
            sampled_batch.on_policy,
            pqt_batch.actions,
            pqt_batch.obs,
            pqt_batch.priors,
            pqt_batch.lengths,
            pqt_batch.rewards,
            pqt_batch.on_policy,
            baseline
        )

        # Store grads and vars for summary
        self._last_grads_and_vars = list(zip(gradients, self.policy.trainable_variables))

        # Build summary dict
        loss_dict = {
            "total_loss": float(total_loss),
            "entropy_loss": float(entropy_loss),
            "pqt_loss": float(pqt_loss)
        }

        # Write summaries if enabled
        if self.summary and self.summary_writer is not None:
            self._write_summaries(iteration, loss_dict, sampled_batch, baseline)
            with self.summary_writer.as_default():
                tf.summary.scalar("pqt_loss", pqt_loss, step=iteration)

        return loss_dict
