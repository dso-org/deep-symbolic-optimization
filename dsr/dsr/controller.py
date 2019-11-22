from functools import partial

import tensorflow as tf
import numpy as np
from scipy import signal
from numba import jit, prange

from dsr.program import Program


class LinearWrapper(tf.contrib.rnn.LayerRNNCell):
    """
    RNNCell wrapper that adds a linear layer to the output.

    See: https://github.com/tensorflow/models/blob/master/research/brain_coder/single_task/pg_agent.py
    """

    def __init__(self, cell, output_size):
        self.cell = cell
        self._output_size = output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(type(self).__name__):
            outputs, state = self.cell(inputs, state, scope=scope)
            logits = tf.layers.dense(outputs, units=self._output_size)

        return logits, state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)
    

class Controller(object):
    """
    Recurrent neural network (RNN) controller used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using REINFORCE with baseline.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: Print shapes and
        number of parameters for each variable.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_units : int
        Number of LSTM units in the RNN's single layer.

    embedding : bool
        Embed each observation?

    embedding_size : int
        Size of embedding for each observation if embedding=True.

    optimizer : str
        Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

    learning_rate : float
        Learning rate for optimizer.

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

    constrain_inv : bool
        Prevent unary function with inverse unary function parent?

    constrain_min_len : bool
        Prevent terminals that would cause the expression to be shorter than
        min_length? If False, only trivial expressions (length 1) are prevented.

    constrain_max_len : bool
        Prevent unary/binary functions that would cause the expression to exceed
        max_length? If False, sampling ends after max_length and dangling nodes
        are filled in with x1's.

    constrain_num_const : bool
        Prevent constants that would exceed max_const?

    min_length : int (>= 1) or None
        Minimum length of a sampled traversal when constrain_min_len=True. If
        None or constrain_min_len=False, expressions have no minimum length.

    max_length : int (>= 3)
        Maximum length of a sampled traversal.

    max_const : int (>= 1) or None
        Maximum number of constants of a sampled traversal when
        constrain_num_const=True. If None or constrain_num_const=False,
        expressions may have any number of constants.

    entropy_weight : float
        Coefficient for entropy bonus.

    ppo : bool
        Use proximal policy optimization (instead of vanilla policy gradient)?

    ppo_clip_ratio : float
        Clip ratio to use for PPO.

    ppo_n_iters : int
        Number of optimization iterations for PPO.

    ppo_n_mb : int
        Number of minibatches per optimization iteration for PPO.

    pqt : bool
        Train with priority queue training (PQT)?

    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

    """

    def __init__(self, sess, debug=0, summary=True,
                 # Architecture hyperparameter
                 # RNN cell hyperparameters
                 cell="lstm",
                 num_units=32,
                 # Embedding hyperparameters
                 embedding=False,
                 embedding_size=4,
                 # Optimizer hyperparameters
                 optimizer='adam',
                 learning_rate=0.001,
                 # Observation space hyperparameters
                 observe_action=True,
                 observe_parent=True,
                 observe_sibling=True,
                 # Constraint hyperparameters
                 constrain_const=True,
                 constrain_trig=True,
                 constrain_inv=True,
                 constrain_min_len=True,
                 constrain_max_len=True,
                 constrain_num_const=False,
                 min_length=2,
                 max_length=30,
                 max_const=None,
                 # Loss hyperparameters
                 entropy_weight=0.0,
                 # PPO hyperparameters
                 ppo=False,
                 ppo_clip_ratio=0.2,
                 ppo_n_iters=10,
                 ppo_n_mb=4,
                 # PQT hyperparameters
                 pqt=False,
                 pqt_k=10,
                 pqt_batch_size=1,
                 pqt_weight=200.0,
                 pqt_use_pg=False):

        self.sess = sess
        self.summary = summary
        self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling

        # Hyperparameters
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.constrain_const = constrain_const and "const" in Program.library.values()
        self.constrain_trig = constrain_trig
        self.constrain_inv = constrain_inv
        self.constrain_min_len = constrain_min_len
        self.constrain_max_len = constrain_max_len
        self.constrain_num_const = constrain_num_const
        self.min_length = min_length
        self.max_length = max_length
        self.max_const = max_const
        self.entropy_weight = entropy_weight
        self.ppo = ppo
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size

        n_choices = Program.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")

        # Parameter assertions/warnings
        assert observe_action + observe_parent + observe_sibling > 0, "Must include at least one observation."
        assert max_length >= 3, "Must have max length at least 3."

        if min_length is None:
            assert not constrain_min_len, "Cannot constrain min length when min_length=None"
        else:
            assert min_length >= 1, "Must have min length at least 1."
            assert max_length >= min_length, "Min length cannot exceed max length."
            if not constrain_min_len:
                print("Warning: min_length={} will not be respected because constrain_min_len=False. Overriding to None.".format(min_length))
                self.min_length = None

        if max_const is None:
            assert not constrain_num_const, "Cannot constrain max num consts when max_const=None"
        else:
            assert max_const >= 1, "Must have max num const at least 1."
            if Program.const_token is None:
                print("Warning: max_const={} will have no effect because there is no constant token.".format(max_const))
                self.constrain_num_const = False
                self.max_const = None
            elif not constrain_num_const:
                print("Warning: max_const={} will not be repsected because constrain_num_const=False. Overriding to None.".format(max_const))
                self.max_const = None

        self.compute_parents_siblings = any([self.observe_parent,
                                             self.observe_sibling,
                                             self.constrain_const])

        # Build controller RNN
        with tf.name_scope("controller"):

            def make_cell(name, num_units, initializer):
                if name == 'lstm':
                    return tf.nn.rnn_cell.LSTMCell(num_units, initializer=initializer)
                if name == 'gru':
                    return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=initializer, bias_initializer=initializer)

            # Create recurrent cell
            cell = make_cell(cell, num_units, initializer=tf.zeros_initializer())
            cell = LinearWrapper(cell=cell, output_size=n_choices)

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
            input_dims = tf.stack([self.batch_size, n_inputs])

            # Create embeddings
            if embedding:
                with tf.variable_scope("embeddings",
                                       initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)):
                    if observe_action:
                        action_embeddings = tf.get_variable("action_embeddings", [n_action_inputs, embedding_size])
                    if observe_parent:
                        parent_embeddings = tf.get_variable("parent_embeddings", [n_parent_inputs, embedding_size])
                    if observe_sibling:
                        sibling_embeddings = tf.get_variable("sibling_embeddings", [n_sibling_inputs, embedding_size])

            # First input is all empty tokens
            observations = []
            if embedding:                
                if observe_action:
                    obs = tf.constant(n_action_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, [self.batch_size])
                    obs = tf.nn.embedding_lookup(action_embeddings, obs)
                    observations.append(obs)
                if observe_parent:
                    obs = tf.constant(n_parent_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, [self.batch_size])
                    obs = tf.nn.embedding_lookup(parent_embeddings, obs)
                    observations.append(obs)
                if observe_sibling:
                    obs = tf.constant(n_sibling_inputs - 1, dtype=tf.int32)
                    obs = tf.broadcast_to(obs, [self.batch_size])
                    obs = tf.nn.embedding_lookup(sibling_embeddings, obs)
                    observations.append(obs)
                cell_input = tf.concat(observations, 1) # Shape (?, n_inputs)
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
                cell_input = tf.broadcast_to(cell_input, input_dims) # Shape (?, n_inputs)

            # Define prior on logits; currently only used to apply hard constraints
            arities = np.array([Program.arities[i] for i in range(n_choices)])
            prior = np.zeros(n_choices, dtype=np.float32)
            if self.min_length is not None and self.min_length > 1:
                prior[arities == 0] = -np.inf
            prior = tf.constant(prior, dtype=tf.float32)
            prior_dims = tf.stack([self.batch_size, n_choices])
            prior = tf.broadcast_to(prior, prior_dims)
            initial_prior = prior


            # Applies constraints
            def get_action_parent_sibling_prior_dangling(actions, dangling):
                n = actions.shape[0] # Batch size
                i = actions.shape[1] - 1 # Current index
                action = actions[:, -1] # Current action

                prior = np.zeros((n, Program.L), dtype=np.float32)

                # Depending on the constraints, may need to compute parents and siblings
                if self.compute_parents_siblings:
                    parent, sibling = parents_siblings(actions, arities=Program.arities_numba, parent_adjust=Program.parent_adjust)
                else:
                    parent = np.zeros(n_parent_inputs, dtype=np.int32)
                    sibling = np.zeros(n_sibling_inputs, dtype=np.int32)

                # Update dangling
                # Fast dictionary lookup of arities for each element in action
                unique, inv = np.unique(action, return_inverse=True)
                dangling += np.array([Program.arities[t] - 1 for t in unique])[inv]

                # Constrain unary of constant or binary of two constants
                if self.constrain_const:
                    # Use action instead of parent here because it's really adj_parent
                    constraints = np.isin(action, Program.unary_tokens) # Unary action (or unary parent)
                    constraints += sibling == Program.const_token # Constant sibling
                    prior += make_prior(constraints, [Program.const_token], Program.L)
                
                # Constrain trig function with trig function ancestor
                if self.constrain_trig:
                    constraints = trig_ancestors(actions, Program.arities_numba, Program.trig_tokens)
                    prior += make_prior(constraints, Program.trig_tokens, Program.L)
                
                # Constrain inverse unary operators
                if self.constrain_inv:
                    for p, c in Program.inverse_tokens.items():
                        # No need to compute parents because only unary operators are constrained
                        # by their inverse, and action == parent for all unary operators
                        constraints = action == p
                        prior += make_prior(constraints, [c], Program.L)
                
                # Constrain total number of constants
                if self.constrain_num_const:
                    constraints = np.sum(actions == Program.const_token, axis=1) == self.max_const
                    prior += make_prior(constraints, [Program.const_token], Program.L)

                # Constrain maximum sequence length
                # Never need to constrain max length for first half of expression
                if self.constrain_max_len and (i + 2) >= self.max_length // 2:
                    remaining = self.max_length - (i + 1)
                    assert sum(dangling > remaining) == 0, (dangling, remaining)
                    constraints = dangling >= remaining - 1 # Constrain binary
                    prior += make_prior(constraints, Program.binary_tokens, Program.L)
                    constraints = dangling == remaining # Constrain unary
                    prior += make_prior(constraints, Program.unary_tokens, Program.L)

                # Constrain minimum sequence length
                # Constrain terminals when dangling == 1 until selecting the (min_length)th token
                if self.constrain_min_len and (i + 2) < self.min_length:
                    constraints = dangling == 1 # Constrain terminals
                    prior += make_prior(constraints, Program.terminal_tokens, Program.L)

                return action, parent, sibling, prior, dangling


            # Given the actions chosen so far, return the next RNN cell input, the prior, and the updated dangling
            # Handles embeddings vs one-hot, observing previous/parent/sibling, and py_func to retrive action/parent/sibling/dangling
            def get_next_input_prior_dangling(actions_ta, dangling):

                # Get current action batch
                actions = tf.transpose(actions_ta.stack()) # Shape: (?, time)
                
                # Compute parent, sibling, prior, and dangling
                action, parent, sibling, prior, dangling = tf.py_func(func=get_action_parent_sibling_prior_dangling,
                                                              inp=[actions, dangling],
                                                              Tout=[tf.int32, tf.int32, tf.int32, tf.float32, tf.int32])

                # Observe previous action, parent, and/or sibling
                observations = []
                if observe_action:
                    if embedding:
                        obs = tf.nn.embedding_lookup(action_embeddings, action)
                    else:
                        obs = tf.one_hot(action, depth=n_action_inputs)
                    observations.append(obs)
                if observe_parent:
                    if embedding:
                        obs = tf.nn.embedding_lookup(parent_embeddings, parent)
                    else:
                        obs = tf.one_hot(parent, depth=n_parent_inputs)
                    observations.append(obs)
                if observe_sibling:
                    if embedding:
                        obs = tf.nn.embedding_lookup(sibling_embeddings, sibling)
                    else:
                        obs = tf.one_hot(sibling, depth=n_sibling_inputs)
                    observations.append(obs)
                input_ = tf.concat(observations, 1)

                # Set the shapes for returned Tensors
                input_.set_shape([None, n_inputs])
                prior.set_shape([None, Program.L])
                dangling.set_shape([None])

                return input_, prior, dangling


            # Define loop function to be used by tf.nn.raw_rnn.
            initial_cell_input = cell_input # Old cell_input defined above
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_output is None: # time == 0
                    finished = tf.zeros(shape=[self.batch_size], dtype=tf.bool)
                    next_input = input_ = initial_cell_input
                    next_cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32) # 2-tuple, each shape (?, num_units)                    
                    emit_output = None
                    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False) # Read twice
                    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    priors_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    prior = initial_prior
                    lengths = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    dangling = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    next_loop_state = (
                        actions_ta,
                        inputs_ta,
                        priors_ta,
                        input_,
                        prior,
                        dangling,
                        lengths, # Unused until implementing variable length
                        finished)
                else:
                    actions_ta, inputs_ta, priors_ta, input_, prior, dangling, lengths, finished = loop_state
                    logits = cell_output + prior
                    next_cell_state = cell_state
                    emit_output = logits
                    action = tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32)[:, 0]
                    # When implementing variable length:
                    # action = tf.where(
                    #     tf.logical_not(finished),
                    #     tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32)[:, 0],
                    #     tf.zeros(shape=[self.batch_size], dtype=tf.int32))
                    next_actions_ta = actions_ta.write(time - 1, action) # Write chosen actions
                    next_input, next_prior, next_dangling = get_next_input_prior_dangling(next_actions_ta, dangling)
                    next_inputs_ta = inputs_ta.write(time - 1, input_) # Write OLD input
                    next_priors_ta = priors_ta.write(time - 1, prior) # Write OLD prior
                    finished = next_finished = tf.logical_or(
                        finished,
                        time >= self.max_length)
                    # When implementing variable length:
                    # finished = next_finished = tf.logical_or(tf.logical_or(
                    #     finished, # Already finished
                    #     next_dangling == 0), # Currently, this will be 0 not just the first time, but also at max_length
                    #     time >= self.max_length)
                    next_lengths = tf.where(
                        finished, # Ever finished
                        lengths,
                        tf.tile(tf.expand_dims(time + 1, 0), [self.batch_size]))
                    next_loop_state = (next_actions_ta,
                                       next_inputs_ta,
                                       next_priors_ta,
                                       next_input,
                                       next_prior,
                                       next_dangling,
                                       next_lengths,
                                       next_finished)

                return (finished, next_input, next_cell_state, emit_output, next_loop_state)

            # Returns RNN emit outputs (TensorArray), final cell state, and final loop state
            with tf.variable_scope('policy'):
                logits_ta, _, loop_state = tf.nn.raw_rnn(cell=cell, loop_fn=loop_fn)
                actions_ta, inputs_ta, priors_ta, _, _, _, lengths, _ = loop_state

            # TBD: Implement a sample class, like PQT?
            self.actions = tf.transpose(actions_ta.stack(), perm=[1, 0]) # (?, max_length)
            self.inputs = tf.transpose(inputs_ta.stack(), perm=[1, 0, 2]) # (?, n_inputs, max_length)
            self.priors = tf.transpose(priors_ta.stack(), perm=[1, 0, 2]) # (?, n_inputs, max_length)
            # self.logits = tf.transpose(logits_ta.stack(), perm=[1, 0, 2]) # (?, max_length)


        # Generates dictionary containing placeholders needed for a batch of sequences
        def make_batch_ph(name):
            with tf.name_scope(name):
                dict_ = {
                    "actions" : tf.placeholder(tf.int32, [None, max_length]),
                    "inputs" : tf.placeholder(tf.float32, [None, max_length, n_inputs]),
                    "priors" : tf.placeholder(tf.float32, [None, max_length, n_choices]),
                    "lengths" : tf.placeholder(tf.int32, [None,]),
                    "masks" : tf.placeholder(tf.float32, [None, max_length])
                }

            return dict_


        # Generates tensor for neglogp of a batch given actions, inputs, priors, masks, and lengths
        def make_neglogp(actions, inputs, priors, masks, lengths):
            with tf.variable_scope('policy', reuse=True):
                logits, _ = tf.nn.dynamic_rnn(cell=cell,
                                              inputs=inputs,
                                              sequence_length=lengths,
                                              dtype=tf.float32)
            logits += priors
            
            # Negative log probabilities of sequences
            neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=actions)
            neglogp = tf.reduce_sum(neglogp_per_step * masks, axis=1) # Sum over time

            # NOTE: The above implementation is the same as the one below, with a few caveats:
            #   Exactly equivalent when removing priors.
            #   Equivalent up to precision when including clipped prior.
            #   Crashes when prior is not clipped due to multiplying zero by -inf.
            # actions_one_hot = tf.one_hot(self.actions_ph, depth=n_choices, axis=-1, dtype=tf.float32)
            # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(self.priors_ph, -2.4e38, 0)) * actions_one_hot
            # neglogp_per_step = tf.reduce_sum(neglogp_per_step, axis=2)
            # neglogp = self.neglogp = tf.reduce_sum(neglogp_per_step * self.mask_ph, axis=1) # Sum over time

            return neglogp, neglogp_per_step


        # On policy batch (used for REINFORCE/PPO)
        self.sampled_batch = make_batch_ph("sampled_batch")

        # Off policy batch (used for PQT)
        if pqt:
            self.off_policy_batch = make_batch_ph("off_policy_batch")

        # Set up losses
        with tf.name_scope("losses"):

            neglogp, neglogp_per_step = make_neglogp(**self.sampled_batch)

            # Entropy loss
            # Entropy = neglogp * p = neglogp * exp(-neglogp)
            entropy_per_step = neglogp_per_step * tf.exp(-neglogp_per_step)
            entropy = tf.reduce_sum(entropy_per_step * self.sampled_batch["masks"], axis=1) # Sum over time
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy, name="entropy_loss")
            loss = entropy_loss

            # PPO loss
            if ppo:
                assert not pqt, "PPO is not compatible with PQT"

                self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="old_neglogp")
                ratio = tf.exp(self.old_neglogp_ph - neglogp)
                clipped_ratio = tf.clip_by_value(ratio, 1. - ppo_clip_ratio, 1. + ppo_clip_ratio)
                ppo_loss = -tf.reduce_mean(tf.minimum(ratio * (self.r - self.baseline), clipped_ratio * (self.r - self.baseline)))
                loss += ppo_loss

                # Define PPO diagnostics
                clipped = tf.logical_or(ratio < (1. - ppo_clip_ratio), ratio > 1. + ppo_clip_ratio)
                self.clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
                self.sample_kl = tf.reduce_mean(neglogp - self.old_neglogp_ph)

            # Policy gradient loss
            else:
                if not pqt or (pqt and pqt_use_pg):
                    pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
                    loss += pg_loss

            # Priority queue training loss
            if pqt:
                pqt_neglogp, _ = make_neglogp(**self.off_policy_batch)
                pqt_loss = pqt_weight * tf.reduce_mean(pqt_neglogp, name="pqt_loss")
                loss += pqt_loss

            self.loss = loss

        # Create summaries
        with tf.name_scope("summary"):
            if self.summary:
                if ppo:
                    tf.summary.scalar("ppo_loss", ppo_loss)
                else:
                    if not pqt or (pqt and pqt_use_pg):
                        tf.summary.scalar("pg_loss", pg_loss)
                if pqt:
                    tf.summary.scalar("pqt_loss", pqt_loss)
                tf.summary.scalar("entropy_loss", entropy_loss)
                tf.summary.scalar("total_loss", self.loss)
                tf.summary.scalar("reward", tf.reduce_mean(self.r))
                tf.summary.scalar("baseline", self.baseline)
                tf.summary.histogram("reward", self.r)
                tf.summary.histogram("length", tf.reduce_sum(self.sampled_batch["masks"], axis=0))
                self.summaries = tf.summary.merge_all()


        def make_optimizer(name, learning_rate):
            if name == "adam":
                return tf.train.AdamOptimizer(learning_rate=learning_rate)
            if name == "rmsprop":
                return tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99)
            if name == "sgd":
                return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


        # Create training op
        optimizer = make_optimizer(name=optimizer, learning_rate=learning_rate)
        with tf.name_scope("train"):
            self.train_op = optimizer.minimize(self.loss)

        if debug >= 1:
            total_parameters = 0
            print("")
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                n_parameters = np.product(shape)
                total_parameters += n_parameters
                print("Variable:    ", variable.name)
                print("  Shape:     ", shape)
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)


    def sample(self, n):
        """Sample batch of n expressions"""

        feed_dict = {self.batch_size : n}

        actions, inputs, priors = self.sess.run([self.actions, self.inputs, self.priors], feed_dict=feed_dict)

        return actions, inputs, priors


    def train_step(self, r, b, actions, inputs, priors, mask, priority_queue):
        """Computes loss, trains model, and returns summaries."""

        feed_dict = {self.r : r,
                     self.baseline : b,
                     self.sampled_batch["actions"] : actions,
                     self.sampled_batch["inputs"] : inputs,
                     self.sampled_batch["lengths"] : np.full(shape=(actions.shape[0]), fill_value=self.max_length, dtype=np.int32),
                     self.sampled_batch["priors"] : priors,
                     self.sampled_batch["masks"] : mask}

        if self.pqt:
            # Sample from the priority queue
            dicts = [extra_data for (item, extra_data) in priority_queue.random_sample(self.pqt_batch_size)]
            pqt_actions = np.stack([d["actions"] for d in dicts], axis=0)
            pqt_inputs = np.stack([d["inputs"] for d in dicts], axis=0)
            pqt_priors = np.stack([d["priors"] for d in dicts], axis=0)
            pqt_masks = np.stack([d["masks"] for d in dicts], axis=0)

            # Update the feed_dict
            feed_dict.update({
                self.off_policy_batch["actions"] : pqt_actions,
                self.off_policy_batch["inputs"] : pqt_inputs,
                self.off_policy_batch["lengths"] : np.full(shape=(pqt_actions.shape[0]), fill_value=self.max_length, dtype=np.int32),
                self.off_policy_batch["priors"] : pqt_priors,
                self.off_policy_batch["masks"] : pqt_masks
                })

        if self.ppo:
            # Compute old_neglogp to be used for training
            old_neglogp = self.sess.run(self.neglogp, feed_dict=feed_dict)

            # Perform multiple epochs of minibatch training
            feed_dict[self.old_neglogp_ph] = old_neglogp
            indices = np.arange(len(r))
            for epoch in range(self.ppo_n_iters):
                self.rng.shuffle(indices)
                minibatches = np.array_split(indices, self.ppo_n_mb)
                for i, mb in enumerate(minibatches):
                    mb_feed_dict = {k : v[mb] for k, v in feed_dict.items() if k not in [self.baseline, self.batch_size, self.sampled_batch["masks"]]}
                    mb_feed_dict.update({
                        self.baseline : b,
                        self.sampled_batch["masks"] : mask[mb, :],
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

    adj_parents : np.ndarray, shape=(N,), dtype=np.int32
        Adjusted parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """

    N, L = tokens.shape
    empty_parent = len(parent_adjust) # Empty token is after all non-empty tokens
    empty_sibling = len(arities) # Empty token is after all non-empty tokens
    adj_parents = np.full(shape=(N,), fill_value=empty_parent, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=empty_sibling, dtype=np.int32)
    # Parallelized loop over action sequences
    for r in prange(N):
        arity = arities[tokens[r, -1]]
        if arity > 0: # Parent is the previous element; no sibling
            adj_parents[r] = parent_adjust[tokens[r, -1]]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-c-1, sibling is the next
                adj_parents[r] = parent_adjust[tokens[r, L - c - 1]]
                siblings[r] = tokens[r, L - c]
                break
    return adj_parents, siblings

