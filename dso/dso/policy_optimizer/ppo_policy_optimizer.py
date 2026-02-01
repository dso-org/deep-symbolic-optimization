
import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.memory import Batch

class PPOPolicyOptimizer(PolicyOptimizer):
    """Proximal policy optimization policy optimizer.

    Parameters
    ----------

    ppo_clip_ratio : float
        Clip ratio to use for PPO.

    ppo_n_iters : int
        Number of optimization iterations for PPO.

    ppo_n_mb : int
        Number of minibatches per optimization iteration for PPO.

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
            # PPO hyperparameters
            ppo_clip_ratio : float = 0.2,
            ppo_n_iters : int = 10,
            ppo_n_mb : int = 4) -> None:
        self.ppo_clip_ratio = ppo_clip_ratio
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.rng = np.random.RandomState(0)  # Used for PPO minibatch sampling
        super()._setup_policy_optimizer(policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)

    def _set_loss(self):
        """Not used in TF2 eager execution - loss is computed in train_step."""
        pass

    def _preppend_to_summary(self):
        """Additional summary fields for PPO."""
        pass

    @tf.function
    def _train_step_tf(self, actions, obs, priors, lengths, rewards, on_policy, old_neglogp, baseline):
        """TF function for training step with gradient tape."""
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

            # PPO loss
            r = tf.cast(rewards, tf.float32)
            baseline_f32 = tf.cast(baseline, tf.float32)
            ratio = tf.exp(old_neglogp - neglogp)
            clipped_ratio = tf.clip_by_value(ratio, 1. - self.ppo_clip_ratio,
                                        1. + self.ppo_clip_ratio)
            ppo_loss = -tf.reduce_mean((r - baseline_f32) *
                                    tf.minimum(ratio, clipped_ratio))

            # Total loss
            total_loss = entropy_loss + ppo_loss

            # PPO diagnostics
            clipped = tf.logical_or(ratio < (1. - self.ppo_clip_ratio),
                                ratio > 1. + self.ppo_clip_ratio)
            clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
            sample_kl = tf.reduce_mean(neglogp - old_neglogp)

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        return total_loss, entropy_loss, ppo_loss, clip_fraction, sample_kl, gradients

    def _compute_neglogp(self, sampled_batch):
        """Compute negative log probabilities for the batch."""
        neglogp, _ = self.policy.make_neglogp_and_entropy(sampled_batch, self.entropy_gamma)
        return neglogp

    def train_step(self, baseline, sampled_batch, iteration=0):
        """Computes loss, trains model, and returns summaries."""
        n_samples = sampled_batch.rewards.shape[0]

        # Compute old_neglogp to be used for training
        old_neglogp = self._compute_neglogp(sampled_batch).numpy()

        # Perform multiple steps of minibatch training
        indices = np.arange(n_samples)
        total_loss = 0
        entropy_loss = 0
        ppo_loss = 0

        for ppo_iter in range(self.ppo_n_iters):
            self.rng.shuffle(indices)  # in-place
            # list of [ppo_n_mb] arrays
            minibatches = np.array_split(indices, self.ppo_n_mb)
            for i, mb in enumerate(minibatches):
                mb_total_loss, mb_entropy_loss, mb_ppo_loss, clip_fraction, sample_kl, gradients = self._train_step_tf(
                    tf.constant(sampled_batch.actions[mb]),
                    tf.constant(sampled_batch.obs[mb]),
                    tf.constant(sampled_batch.priors[mb]),
                    tf.constant(sampled_batch.lengths[mb]),
                    tf.constant(sampled_batch.rewards[mb]),
                    tf.constant(sampled_batch.on_policy[mb]),
                    tf.constant(old_neglogp[mb]),
                    baseline
                )
                total_loss = mb_total_loss
                entropy_loss = mb_entropy_loss
                ppo_loss = mb_ppo_loss

        # Store grads and vars for summary
        self._last_grads_and_vars = list(zip(gradients, self.policy.trainable_variables))

        # Build summary dict
        loss_dict = {
            "total_loss": float(total_loss),
            "entropy_loss": float(entropy_loss),
            "ppo_loss": float(ppo_loss)
        }

        # Write summaries if enabled
        if self.summary and self.summary_writer is not None:
            self._write_summaries(iteration, loss_dict, sampled_batch, baseline)
            with self.summary_writer.as_default():
                tf.summary.scalar("ppo_loss", ppo_loss, step=iteration)

        return loss_dict
