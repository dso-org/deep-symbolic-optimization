import tensorflow as tf
import numpy as np
from scipy import signal
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
    
    constrain_const : bool
        Prevent constants with unary parents or constant siblings?

    constrain_trig : bool
        Prevent trig functions with trig function ancestors?

    ppo : bool
        Use proximal policy optimization (instead of vanilla policy gradient)?

    ppo_clip_ratio : float
        Clip ratio to use for PPO.

    ppo_n_iters : int
        Number of optimization iterations for PPO.

    ppo_n_mb : int
        Number of minibatches per optimization iteration for PPO.

    embedding : bool
        Embed each observation?

    embedding_size : int
        Size of embedding for each observation if embedding=True.
    """

    def __init__(self, sess, num_units, max_length, learning_rate=0.001,
                 entropy_weight=0.0, observe_action=True, observe_parent=True,
                 observe_sibling=True, summary=True, constrain_const=True,
                 constrain_trig=True, ppo=False, ppo_clip_ratio=0.2,
                 ppo_n_iters=10, ppo_n_mb=4, embedding=False, embedding_size=4):

        self.sess = sess
        self.actions = [] # Actions sampled from the controller
        self.logits = []
        self.summary = summary
        self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.max_length = max_length
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.constrain_const = constrain_const and "const" in Program.library
        self.constrain_trig = constrain_trig
        self.ppo = ppo
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb

        # Buffers from previous batch
        self.prev_parents = []
        self.prev_siblings = []
        self.prev_priors = []

        neglogps = []
        entropies = []
        n_choices = Program.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.actions_ph = []
        self.parents_ph = []
        self.siblings_ph = []
        self.priors_ph = []
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        self.actions_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, None), name="actions_mask")

        # Parameter assertions
        assert observe_action + observe_parent + observe_sibling > 0, "Must include at least one observation."

        self.compute_parents_siblings = any([self.observe_parent,
                                             self.observe_sibling,
                                             self.constrain_const])
        self.compute_prior = any([self.constrain_const,
                                  self.constrain_trig])

        # Build controller RNN
        with tf.name_scope("controller"):

            # Create LSTM cell
            cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.zeros_initializer())
            cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32) # 2-tuple, each shape (?, num_units)

            # Define input dimensions
            n_action_inputs = n_choices + 1 # Library tokens + empty token
            n_parent_inputs = n_choices + 1 - len(Program.terminal_tokens) # Parent sub-library tokens + empty token
            n_sibling_inputs = n_choices + 1 # Library tokens + empty tokens
            if embedding:
                n_inputs = observe_action * embedding_size + \
                           observe_parent * embedding_size + \
                           observe_sibling * embedding_size
            else:
                n_inputs = observe_action * n_action_inputs + \
                           observe_parent * n_parent_inputs + \
                           observe_sibling * n_sibling_inputs
            input_dims = tf.stack([self.batch_size, 1, n_inputs])

            # Create embeddings
            if embedding:
                if observe_action:
                    action_embeddings = tf.get_variable("action_embeddings", [n_action_inputs, embedding_size])
                if observe_parent:
                    parent_embeddings = tf.get_variable("parent_embeddings", [n_parent_inputs, embedding_size])
                if observe_sibling:
                    sibling_embeddings = tf.get_variable("sibling_embeddings", [n_sibling_inputs, embedding_size])

            # First input is all empty tokens
            observations = [] # Each observation has shape (?, 1, n_choices + 1) or (?, 1, embedding_size)
            if embedding:                
                if observe_action:
                    obs = tf.constant(n_action_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, tf.stack([self.batch_size, 1]))
                    obs = tf.nn.embedding_lookup(action_embeddings, obs)
                    observations.append(obs)
                if observe_parent:
                    obs = tf.constant(n_parent_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, tf.stack([self.batch_size, 1]))
                    obs = tf.nn.embedding_lookup(parent_embeddings, obs)
                    observations.append(obs)
                if observe_sibling:
                    obs = tf.constant(n_sibling_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, tf.stack([self.batch_size, 1]))
                    obs = tf.nn.embedding_lookup(sibling_embeddings, obs)
                    observations.append(obs)
                cell_input = tf.concat(observations, 2) # Shape (?, 1, n_inputs)
            else:
                observations = []
                if observe_action:
                    obs = [0]*(n_action_inputs)
                    obs[n_action_inputs - 1] = 1
                    observations += obs
                if observe_parent:
                    obs = [0]*(n_parent_inputs)
                    obs[n_parent_inputs - 1] = 1
                    observations += obs
                if observe_sibling:
                    obs = [0]*(n_sibling_inputs)
                    obs[n_sibling_inputs - 1] = 1
                    observations += obs
                observations = np.array(observations, dtype=np.float32)
                cell_input = tf.constant(observations)
                cell_input = tf.broadcast_to(cell_input, input_dims) # Shape (?, 1, n_inputs)

            # Define prior on logits; currently only used to apply hard constraints
            # First node must be nonterminal
            arities = np.array([Program.arities[i] for i in range(n_choices)])
            prior = np.zeros(len(arities), dtype=np.float32)
            prior[arities == 0] = -np.inf
            prior = tf.constant(prior, dtype=tf.float32)

            for i in range(max_length):
                outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                        cell_input,
                                                        initial_state=cell_state,
                                                        dtype=tf.float32)

                # Outputs correspond to logits of library
                logits = tf.layers.dense(outputs[:, -1, :], units=n_choices, reuse=tf.AUTO_REUSE)
                logits = logits + prior
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
                # Must be three dimensions: [batch_size, sequence_length, n_inputs]
                observations = [] # Each observation has shape (?, 1, n_choices + 1) or (?, 1, embedding_size)
                if observe_action:
                    ph = tf.reshape(action_ph, (self.batch_size, 1))
                    if embedding:
                        obs = tf.nn.embedding_lookup(action_embeddings, ph)                        
                    else:
                        obs = tf.one_hot(ph, depth=n_action_inputs)
                    observations.append(obs)
                if observe_parent:
                    parent_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                    self.parents_ph.append(parent_ph)
                    ph = tf.reshape(parent_ph, (self.batch_size, 1))
                    if embedding:
                        obs = tf.nn.embedding_lookup(parent_embeddings, ph)
                    else:
                        obs = tf.one_hot(ph, depth=n_parent_inputs)
                    observations.append(obs)
                if observe_sibling:
                    sibling_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                    self.siblings_ph.append(sibling_ph)
                    ph = tf.reshape(sibling_ph, (self.batch_size, 1))
                    if embedding:
                        obs = tf.nn.embedding_lookup(sibling_embeddings, ph)
                    else:
                        obs = tf.one_hot(ph, depth=n_sibling_inputs)
                    observations.append(obs)
                cell_input = tf.concat(observations, 2, name="cell_input_{}".format(i)) # Shape: (?, 1, n_inputs)

                # Update LSTM state
                cell_state = final_state

                # Update prior
                if self.compute_prior:
                    prior_ph = tf.placeholder(dtype=tf.float32, shape=(None, n_choices))
                    self.priors_ph.append(prior_ph)
                    prior = prior_ph
                else:
                    prior = tf.zeros_like(prior)

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

            # Entropy loss is negative entropy, since entropy provides a bonus
            entropy_loss = -self.entropy_weight*tf.reduce_mean(self.sample_entropy, name="entropy_loss")

            if ppo:

                # Define PPO loss
                self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="old_neglogp")
                ratio = tf.exp(self.old_neglogp_ph - self.sample_neglogp)
                clipped_ratio = tf.clip_by_value(ratio, 1. - ppo_clip_ratio, 1. + ppo_clip_ratio)
                ppo_loss = -tf.reduce_mean(tf.minimum(ratio * (self.r - self.baseline), clipped_ratio * (self.r - self.baseline)))
                self.loss = ppo_loss + entropy_loss

                # Define PPO diagnostics
                clipped = tf.logical_or(ratio < (1. - ppo_clip_ratio), ratio > 1. + ppo_clip_ratio)
                self.clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
                self.sample_kl = tf.reduce_mean(self.sample_neglogp - self.old_neglogp_ph)

            else:

                # Define VPG loss
                policy_gradient_loss = tf.reduce_mean((self.r - self.baseline) * self.sample_neglogp, name="policy_gradient_loss")
                self.loss = policy_gradient_loss + entropy_loss

        # Create summaries
        if self.summary:
            if ppo:
                tf.summary.scalar("ppo_loss", ppo_loss)
            else:
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
        parents = []
        siblings = []
        priors = []
        feed_dict = {self.batch_size : n}
        for i in range(self.max_length):
            action = self.sess.run(self.actions[i], feed_dict=feed_dict) # Shape: (n,)
            actions.append(action)
            feed_dict[self.actions_ph[i]] = action

            if self.compute_parents_siblings or self.constrain_trig:
                tokens = np.stack(actions).T

            # Compute parents and siblings
            if self.compute_parents_siblings:
                tokens = np.stack(actions).T # Shape: (n, i)
                parent, sibling = parents_siblings(tokens, Program.arities_numba, Program.parent_adjust)
                parents.append(parent)
                siblings.append(sibling)
                if self.observe_parent:
                    feed_dict[self.parents_ph[i]] = parent # Shape: (n,)
                if self.observe_sibling:
                    feed_dict[self.siblings_ph[i]] = sibling # Shape: (n,)

            # Compute prior
            if self.compute_prior:
                prior = np.zeros((n, Program.L), dtype=np.float32)
                if self.constrain_const:
                    constraints = np.isin(parent, Program.unary_tokens) # Unary parent (or unary action)
                    constraints += sibling == Program.const_token # Constant sibling
                    prior += make_prior(constraints, [Program.const_token], Program.L)
                if self.constrain_trig:
                    constraints = trig_ancestors(tokens, Program.arities_numba, Program.trig_tokens)
                    prior += make_prior(constraints, Program.trig_tokens, Program.L)
                feed_dict[self.priors_ph[i]] = prior
                priors.append(prior)

        # Record previous parents, siblings and priors
        # Note the first axis is along length, since they are used in zip to feed placeholders
        if self.compute_parents_siblings:
            self.prev_parents = np.stack(parents) # Shape: (max_length, n)
            self.prev_siblings = np.stack(siblings) # Shape: (max_length, n)
        if self.compute_prior:
            self.prev_priors = np.stack(priors) # Shape: (max_length, n, Program.L)
        
        actions = np.stack(actions).T # Shape: (n, max_length)

        return actions


    # ANTIQUATED
    # def neglogp(self, actions, actions_mask):
    #     """Returns neglogp of batch of expressions"""

    #     feed_dict = {self.actions_ph[i] : a for i,a in enumerate(actions.T)}
    #     feed_dict[self.actions_mask] = actions_mask
    #     feed_dict[self.batch_size] = actions.shape[0]

    #     return self.sess.run(self.sample_neglogp, feed_dict=feed_dict)


    def train_step(self, r, b, actions, actions_mask, i_mask):
        """Computes loss, trains model, and returns summaries."""

        feed_dict = {self.r : r,
                     self.baseline : b,
                     self.actions_mask : actions_mask,
                     self.batch_size : actions.shape[0]}

        # Select the corresponding subsets for parents, siblings, and priors
        if i_mask is not None:
            if self.compute_parents_siblings:
                self.prev_parents = self.prev_parents[:, i_mask]
                self.prev_siblings = self.prev_siblings[:, i_mask]
            if self.compute_prior:
                self.prev_priors = self.prev_priors[:, i_mask, :]
        
        # Zip along trajectory axis
        feed_dict.update(zip(self.actions_ph, actions.T))
        feed_dict.update(zip(self.parents_ph, self.prev_parents))
        feed_dict.update(zip(self.siblings_ph, self.prev_siblings))
        feed_dict.update(zip(self.priors_ph, self.prev_priors))

        if self.ppo:
            # Compute old_neglogp to be used for training
            old_neglogp = self.sess.run(self.sample_neglogp, feed_dict=feed_dict)

            # Perform multiple epochs of minibatch training
            feed_dict[self.old_neglogp_ph] = old_neglogp
            indices = np.arange(len(r))
            for epoch in range(self.ppo_n_iters):
                self.rng.shuffle(indices)
                minibatches = np.array_split(indices, self.ppo_n_mb)
                for i, mb in enumerate(minibatches):
                    mb_feed_dict = {k : v[mb] for k, v in feed_dict.items() if k not in [self.baseline, self.batch_size, self.actions_mask]}
                    mb_feed_dict.update({
                        self.baseline : b,
                        self.actions_mask : actions_mask[:, mb],
                        self.batch_size : len(mb)
                        })

                    _ = self.sess.run([self.train_op], feed_dict=mb_feed_dict)

                    # Diagnostics
                    # kl, cf, _ = self.sess.run([self.sample_kl, self.clip_fraction, self.train_op], feed_dict=mb_feed_dict)
                    # print("epoch", epoch, "i", i, "KL", kl, "CF", cf)

        else:
            _ = self.sess.run([self.train_op], feed_dict=feed_dict)

        # Return summaries
        if self.summary:
            summaries = self.sess.run(self.summaries, feed_dict=feed_dict)
        else:
            summaries = None
        
        return summaries


def make_prior(constraints, constraint_tokens, library_length):
    """
    Given a batch of constraints and the corresponding tokens to be constrained,
    returns a prior that is added to the logits when sampling the next action.

    For example, given library_length=5 and constraint_tokens=[1,2], a
    constrained row of the prior will be: [0.0, -np.inf, -np.inf, 0.0, 0.0].

    Parameters
    __________

    constraints : np.ndarray, shape=(batch_size,), dtype=np.bool_
        Batch of constraints.

    constraint_tokens : np.ndarray, dtype=np.int32
        Array of which tokens to constrain.

    library_length : int
        Length of library.

    Returns
    _______

    prior : np.ndarray, shape=(batch_size, library_length), dtype=np.float32
        Prior adjustment to logits given constraints. Since these are hard
        constraints, ach element is either 0.0 or -np.inf.
    """

    prior = np.zeros((constraints.shape[0], library_length), dtype=np.float32)
    for t in constraint_tokens:
        prior[constraints == True, t] = -np.inf
    return prior


@jit(nopython=True, parallel=True)
def trig_ancestors(tokens, arities, trig_tokens):
    """
    Given a batch of action sequences, determines whether the next element of
    the sequence has an ancestor that is a trigonometric function.
    
    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because they will be masked in loss calculations.

    Parameters
    __________

    tokens : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : numba.typed.Dict
        Dictionary from library index to arity.

    trig_tokens : np.ndarray, dtype=np.int32
        Array of tokens corresponding to trig functions.

    Returns
    _______

    ancestors : np.ndarray, shape=(N,), dtype=np.bool_
        Whether the next element of each sequence has a trig function ancestor.
    """

    N, L = tokens.shape
    ancestors = np.zeros(shape=(N,), dtype=np.bool_)
    # Parallelized loop over action sequences
    for r in prange(N):
        dangling = 0
        threshold = None # If None, current branch does not have trig ancestor
        for c in range(L):
            arity = arities[tokens[r, c]]
            dangling += arity - 1
            # Turn "on" if a trig function is found
            # Remain "on" until branch completes
            if threshold is None:
                for trig_token in trig_tokens:
                    if tokens[r, c] == trig_token:
                        threshold = dangling - 1
                        break
            # Turn "off" once the branch completes
            else:                
                if dangling == threshold:
                    threshold = None
        # If the sequences ended "on", then there is a trig ancestor
        if threshold is not None:
            ancestors[r] = True
    return ancestors

@jit(nopython=True, parallel=True)
def parents_siblings(tokens, arities, parent_adjust):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings of the next element of the sequence.

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because they will be masked in loss calculations.

    Parameters
    __________

    tokens : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : numba.typed.Dict
        Dictionary from library index to arity.

    parent_adjust : numba.typed.Dict
        Dictionary from library index to parent sub-library index.

    Returns
    _______

    parents : np.ndarray, shape=(N,), dtype=np.int32
        Parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """

    N, L = tokens.shape
    empty_parent = len(parent_adjust) # Empty token is after all non-empty tokens
    empty_sibling = len(arities) # Empty token is after all non-empty tokens
    parents = np.full(shape=(N,), fill_value=empty_parent, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=empty_sibling, dtype=np.int32)
    # Parallelized loop over action sequences
    for r in prange(N):
        arity = arities[tokens[r, -1]]
        if arity > 0: # Parent is the previous element; no sibling
            parents[r] = parent_adjust[tokens[r, -1]]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-c-1, sibling is the next
                parents[r] = parent_adjust[tokens[r, L - c - 1]]
                siblings[r] = tokens[r, L - c]
                break
    return parents, siblings

