import tensorflow as tf
import numpy as np
from numba import jit, prange

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

    observe_action : bool
        Observe previous action token?

    observe_parent : bool
        Observe parent token?

    observe_sibling : bool
        Observe sibling token?
    """

    def __init__(self, sess, num_units, max_length, learning_rate=0.001,
                 entropy_weight=0.0, observe_action=True, observe_parent=True,
                 observe_sibling=True):

        self.sess = sess
        self.actions = [] # Actions sampled from the controller
        self.logits = []

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.max_length = max_length
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        neglogps = []
        entropies = []

        n_choices = len(Program.arities)

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.actions_ph = []
        self.parents_ph = []
        self.siblings_ph = []

        assert observe_action + observe_parent + observe_sibling > 0, "Must include at least one observation."

        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        self.actions_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, None), name="actions_mask")

        # Build controller RNN
        with tf.name_scope("controller"):

            cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.zeros_initializer())
            cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            n_observations = observe_action + observe_parent + observe_sibling
            input_dims = tf.stack([self.batch_size, 1, n_observations*n_choices])

            # TBD: Should probably be -1 encoding since that's the no-parent and no-sibling token
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

                # Update LSTM input
                # Must be three dimensions: [batch_size, sequence_length, n_observations*n_choices]
                observations = [] # Each observation has shape : (?, 1, n_choices)
                if observe_action:
                    new_obs = tf.one_hot(tf.reshape(action_ph, (self.batch_size, 1)), depth=n_choices)
                    observations.append(new_obs)
                if observe_parent:
                    parent_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                    self.parents_ph.append(parent_ph)
                    new_obs = tf.one_hot(tf.reshape(parent_ph, (self.batch_size, 1)), depth=n_choices)
                    observations.append(new_obs)
                if observe_sibling:
                    sibling_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                    self.siblings_ph.append(sibling_ph)
                    new_obs = tf.one_hot(tf.reshape(sibling_ph, (self.batch_size, 1)), depth=n_choices)
                    observations.append(new_obs)

                cell_input = tf.concat(observations, 2, name="cell_input_{}".format(i)) # Shape: (?, 1, n_observations*n_choices)

                # Update LSTM state
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
            action = self.sess.run(self.actions[i], feed_dict=feed_dict) # Shape: (n,)
            actions.append(action)
            feed_dict[self.actions_ph[i]] = action

            if self.observe_parent or self.observe_sibling:
                all_actions = np.stack(actions) # Shape: (i, n)
                parents, siblings = parents_siblings(all_actions, Program.arities_numba)
                if self.observe_parent:
                    feed_dict[self.parents_ph[i]] = parents # Shape: (n,)
                if self.observe_sibling:
                    feed_dict[self.siblings_ph[i]] = siblings # Shape: (n,)

        return actions



    def neglogp(self, actions, actions_mask):
        """Returns neglogp of batch of expressions"""

        feed_dict = {self.actions_ph[i] : a for i,a in enumerate(actions.T)}
        feed_dict[self.actions_mask] = actions_mask
        feed_dict[self.batch_size] = actions.shape[0]

        return self.sess.run(self.sample_neglogp, feed_dict=feed_dict)


    def train_step(self, r, b, actions, actions_mask):
        """Computes loss, applies gradients, and computes summaries."""

        feed_dict = {self.r : r,
                    self.baseline : b,
                    self.actions_mask : actions_mask,
                    self.batch_size : actions.shape[0]}

        all_actions = []
        for i, action in enumerate(actions.T):
            feed_dict[self.actions_ph[i]] = action
            all_actions.append(action)

            # TBD: Why does parents_siblings() have to be recalculated? It's not a function of the loss...
            if self.observe_parent or self.observe_sibling:
                tokens = np.stack(all_actions)
                parents, siblings = parents_siblings(tokens, Program.arities_numba)
                if self.observe_parent:
                    feed_dict[self.parents_ph[i]] = parents
                if self.observe_sibling:
                    feed_dict[self.siblings_ph[i]] = siblings

        loss, _, summaries = self.sess.run([self.loss, self.train_op, self.summaries], feed_dict=feed_dict)
        return loss, summaries


@jit(nopython=True, parallel=True)
def parents_siblings(tokens, arities):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings of the last element of the sequence.

    The batch has shape (L, N), where L is the length of each sequence and N is
    the number of sequences (i.e. batch size). In some cases, expressions may
    already be complete; in these cases, this function will see no expression at
    all (parent = -1, sibling = -1, or the start of a new expressions, which may
    have any parent/sibling combination. However, the return value for these
    elements doesn't matter because they will be masked in all loss calculations
    anyway.

    Parameters
    __________

    tokens : np.ndarray, shape=(L, N), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : numba.typed.Dict
        Dictionary from library index to arity.

    Returns
    _______

    parents : np.ndarray, shape=(N,), dtype=np.int32
        Parents of the last element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the last element of each action sequence.

    """

    L, N = tokens.shape
    parents = np.full(shape=(N,), fill_value=-1, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=-1, dtype=np.int32)
    # Parallelized loop over action sequences
    for c in prange(N):
        arity = arities[tokens[-1, c]]
        if arity > 0: # Parent is the previous element; no sibling
            parents[c] = tokens[-1, c]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for r in range(L):
            arity = arities[tokens[L - r - 1, c]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-r-1, sibling is the next
                parents[c] = tokens[L - r - 1, c]
                siblings[c] = tokens[L - r, c]
                break

    return parents, siblings

