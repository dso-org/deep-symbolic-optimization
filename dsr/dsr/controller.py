import tensorflow as tf
import numpy as np

from dsr.program import Program


class Controller(object):
    """
    Recurrent neural network (RNN) controller used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using REINFORCE with baseline.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.

    num_units : int
        Number of LSTM units in the RNN's single layer.

    max_length : int
        Maximum length of a sampled traversal.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.
    """

    def __init__(self, sess, num_units, max_length, learning_rate=0.001,
                 entropy_weight=0.0):

        self.sess = sess
        self.actions = [] # Actions sampled from the controller
        self.logits = []

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.max_length = max_length

        neglogps = []
        entropies = []

        n_choices = len(Program.arities)

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.actions_ph = []
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        self.actions_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, None), name="actions_mask")

        # Build controller RNN
        with tf.name_scope("controller"):
            
            cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.zeros_initializer())
            cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            input_dims = tf.stack([self.batch_size, 1, n_choices])
            cell_input = tf.fill(input_dims, 1.0) # First input fed to controller
            
            #####
            # TBD: Create embedding layer
            #####

            for i in range(max_length):
                ouputs, final_state = tf.nn.dynamic_rnn(cell,
                                                        cell_input,
                                                        initial_state=cell_state,
                                                        dtype=tf.float32)

                # Outputs correspond to logits of library
                logits = tf.layers.dense(ouputs[:, -1, :], units=n_choices)
                if i == 0:
                    # First node must be nonterminal, so set logits to -inf
                    arities = np.array([Program.arities[i] for i in range(n_choices)])
                    adjustment = np.zeros(len(arities), dtype=np.float32)
                    adjustment[arities == 0] = -np.inf
                    adjustment = tf.constant(adjustment, dtype=tf.float32)
                    logits = logits + adjustment
                self.logits.append(logits)

                # Sample from the library
                action = tf.multinomial(logits, num_samples=1)
                action = tf.to_int32(action)
                action = tf.reshape(action, (self.batch_size,))
                self.actions.append(action)

                # Placeholder for selected actions
                action_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                self.actions_ph.append(action_ph)

                # Update LSTM input and state with selected actions
                cell_input = tf.one_hot(tf.reshape(self.actions_ph[i], (self.batch_size, 1)), depth=n_choices, name="cell_input_{}".format(i))
                cell_state = final_state

                # Cross-entropy loss is equivalent to neglogp
                neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[i],
                                                                        labels=action_ph)
                neglogps.append(neglogp)

                # Entropy = neglogp * p = neglogp * exp(-neglogp)
                entropy = neglogp * tf.exp(-neglogp)
                entropies.append(entropy)

            # Reduce neglogps
            neglogps = tf.stack(neglogps) * self.actions_mask # Shape: (max_length, batch_size)
            self.sample_neglogp = tf.reduce_sum(neglogps, axis=0)

            # Reduce entropies
            entropies = tf.stack(entropies) * self.actions_mask # Shape: (max_length, batch_size)
            self.sample_entropy = tf.reduce_sum(entropies, axis=0)

        # Setup losses
        with tf.name_scope("losses"):
            
            # Policy gradient loss is neglogp(actions) scaled by reward
            policy_gradient_loss = tf.reduce_mean((self.r - self.baseline) * self.sample_neglogp, name="policy_gradient_loss")

            # Entropy loss is negative entropy, since entropy provides a bonus
            entropy_loss = -self.entropy_weight*tf.reduce_mean(self.sample_entropy, name="entropy_loss")

            self.loss = policy_gradient_loss + entropy_loss # May add additional terms later

        # Create summaries
        tf.summary.scalar("policy_gradient_loss", policy_gradient_loss)
        tf.summary.scalar("entropy_loss", entropy_loss)
        tf.summary.scalar("total_loss", self.loss)
        tf.summary.scalar("reward", tf.reduce_mean(self.r))
        tf.summary.scalar("baseline", self.baseline)
        tf.summary.histogram("reward", self.r)
        tf.summary.histogram("length", tf.reduce_sum(self.actions_mask, axis=0))
        self.summaries = tf.summary.merge_all()

        # Create training op
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


    def sample(self, n):
        """Sample batch of n expressions"""

        actions = []

        feed_dict = {self.batch_size : n}
        for i in range(self.max_length):
            action = self.sess.run(self.actions[i], feed_dict=feed_dict)
            actions.append(action)
            feed_dict[self.actions_ph[i]] = action

        return actions


    def neglogp(self, actions, actions_mask):
        """Returns neglogp of batch of expressions"""

        feed_dict = {self.actions_ph[i] : a for i,a in enumerate(actions.T)}
        feed_dict[self.actions_mask] = actions_mask
        feed_dict[self.batch_size] = actions.shape[0]

        return self.sess.run(self.sample_neglogp, feed_dict=feed_dict)


    def train_step(self, r, b, actions, actions_mask):
        """Computes loss, applies gradients, and computes summaries"""

        feed_dict = {self.r : r,
                    self.baseline : b,
                    self.actions_mask : actions_mask,
                    self.batch_size : actions.shape[0]}

        for i, action in enumerate(actions.T):
            feed_dict[self.actions_ph[i]] = action

        loss, _, summaries = self.sess.run([self.loss, self.train_op, self.summaries], feed_dict=feed_dict)
        
        return loss, summaries
