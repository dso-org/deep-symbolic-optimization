"""Controller used to generate distribution over hierarchical, variable-length objects."""

import tensorflow as tf
import numpy as np

from dsr.program import Program
from dsr.memory import Batch
from dsr.subroutines import parents_siblings
from dsr.prior import LengthConstraint


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

    prior : dsr.prior.JointPrior
        JointPrior object used to adjust probabilities during sampling.

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: Print shapes and
        number of parameters for each variable.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

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

    max_length : int or None
        Maximum sequence length. This will be overridden if a LengthConstraint
        with a maximum length is part of the prior.

    """

    def __init__(self, sess, prior, debug=0, summary=True,
                 # RNN cell hyperparameters
                 cell='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros',
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
                 pqt_use_pg=False,
                 # Other hyperparameters
                 max_length=None):

        self.sess = sess
        self.prior = prior
        self.summary = summary
        self.rng = np.random.RandomState(0) # Used for PPO minibatch sampling

        lib = Program.library

        # Find max_length from the LengthConstraint prior, if it exists
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break
        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if "\
                "there is no LengthConstraint."
            self.max_length = max_length
            print("WARNING: Maximum length not constrained. Sequences will "
                  "stop at {} and complete by repeating the first input "
                  "variable.".format(self.max_length))
        elif max_length is not None and max_length != self.max_length:
            print("WARNING: max_length ({}) will be overridden by value from "
                  "LengthConstraint ({}).".format(max_length, self.max_length))
        max_length = self.max_length

        # Hyperparameters
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.entropy_weight = entropy_weight
        self.ppo = ppo
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size

        n_choices = lib.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        
        # Parameter assertions/warnings
        assert observe_action + observe_parent + observe_sibling > 0, "Must include at least one observation."

        self.compute_parents_siblings = any([self.observe_parent,
                                             self.observe_sibling,
                                             self.prior.requires_parents_siblings])

        # Build controller RNN
        with tf.name_scope("controller"):

            def make_initializer(name):
                if name == "zeros":
                    return tf.zeros_initializer()
                if name == "var_scale":
                    return tf.contrib.layers.variance_scaling_initializer(
                            factor=0.5, mode='FAN_AVG', uniform=True, seed=0)
                raise ValueError("Did not recognize initializer '{}'".format(name))

            def make_cell(name, num_units, initializer):
                if name == 'lstm':
                    return tf.nn.rnn_cell.LSTMCell(num_units, initializer=initializer)
                if name == 'gru':
                    return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=initializer, bias_initializer=initializer)
                raise ValueError("Did not recognize cell type '{}'".format(name))

            # Create recurrent cell
            if isinstance(num_units, int):
                num_units = [num_units] * num_layers
            initializer = make_initializer(initializer)
            cell = tf.contrib.rnn.MultiRNNCell(
                    [make_cell(cell, n, initializer=initializer) for n in num_units])
            cell = LinearWrapper(cell=cell, output_size=n_choices)

            # Define input dimensions
            n_action_inputs = n_choices + 1 # lib tokens + empty token
            n_parent_inputs = n_choices + 1 - len(lib.terminal_tokens) # Parent sub-lib tokens + empty token
            n_sibling_inputs = n_choices + 1 # lib tokens + empty tokens

            # Create embeddings
            if embedding:
                with tf.variable_scope("embeddings",
                                       initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=0)):
                    if observe_action:
                        action_embeddings = tf.get_variable("action_embeddings", [n_action_inputs, embedding_size], trainable=True)
                    if observe_parent:
                        parent_embeddings = tf.get_variable("parent_embeddings", [n_parent_inputs, embedding_size], trainable=True)
                    if observe_sibling:
                        sibling_embeddings = tf.get_variable("sibling_embeddings", [n_sibling_inputs, embedding_size], trainable=True)

            # First observation is all empty tokens
            initial_obs = tuple()
            for n in [n_action_inputs, n_parent_inputs, n_sibling_inputs]:
                obs = tf.constant(n - 1, dtype=np.int32)
                obs = tf.broadcast_to(obs, [self.batch_size])
                initial_obs += (obs,)            

            # Get initial prior
            initial_prior = self.prior.initial_prior()
            initial_prior = tf.constant(initial_prior, dtype=tf.float32)
            prior_dims = tf.stack([self.batch_size, n_choices])
            initial_prior = tf.broadcast_to(initial_prior, prior_dims)
            # arities = np.array([Program.arities[i] for i in range(n_choices)])
            # prior = np.zeros(n_choices, dtype=np.float32)
            # if self.min_length is not None and self.min_length > 1:
            #     prior[arities == 0] = -np.inf
            # prior = tf.constant(prior, dtype=tf.float32)
            # prior_dims = tf.stack([self.batch_size, n_choices])
            # prior = tf.broadcast_to(prior, prior_dims)
            # initial_prior = prior


            # Returns concatenated one-hot or embeddings from observation tokens
            # Used for both raw_rnn and dynamic_rnn
            def get_input(obs):
                action, parent, sibling = obs
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
                input_ = tf.concat(observations, -1)                
                return input_


            # Applies constraints
            def get_action_parent_sibling_prior_dangling(actions, dangling):
                n = actions.shape[0] # Batch size
                i = actions.shape[1] - 1 # Current index
                action = actions[:, -1] # Current action

                # Depending on the constraints, may need to compute parents and siblings
                if self.compute_parents_siblings:
                    parent, sibling = parents_siblings(actions, arities=lib.arities, parent_adjust=lib.parent_adjust)
                else:
                    parent = np.zeros(n, dtype=np.int32)
                    sibling = np.zeros(n, dtype=np.int32)

                # Update dangling with (arity - 1) for each element in action
                dangling += lib.arities[action] - 1

                prior = self.prior(actions, parent, sibling, dangling)

                return action, parent, sibling, prior, dangling


            # Given the actions chosen so far, return the observation, the prior, and the updated dangling
            # Uses py_func to retrieve action/parent/sibling/dangling
            def get_next_obs_prior_dangling(actions_ta, dangling):

                # Get current action batch
                actions = tf.transpose(actions_ta.stack()) # Shape: (?, time)
                
                # Compute parent, sibling, prior, and dangling
                action, parent, sibling, prior, dangling = tf.py_func(func=get_action_parent_sibling_prior_dangling,
                                                              inp=[actions, dangling],
                                                              Tout=[tf.int32, tf.int32, tf.int32, tf.float32, tf.int32])

                # Observe previous action, parent, and/or sibling
                obs = (action, parent, sibling)

                # Set the shapes for returned Tensors
                action.set_shape([None])
                parent.set_shape([None])
                sibling.set_shape([None])                
                prior.set_shape([None, lib.L])
                dangling.set_shape([None])

                return obs, prior, dangling


            # Define loop function to be used by tf.nn.raw_rnn.
            initial_cell_input = get_input(initial_obs)
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_output is None: # time == 0
                    finished = tf.zeros(shape=[self.batch_size], dtype=tf.bool)
                    obs = initial_obs
                    next_input = get_input(obs)
                    next_cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32) # 2-tuple, each shape (?, num_units)                    
                    emit_output = None
                    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False) # Read twice
                    obs_tas = (tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True), # Action inputs
                              tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True), # Parent inputs
                              tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=True)) # Sibling inputs
                    priors_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    prior = initial_prior
                    lengths = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    dangling = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    next_loop_state = (
                        actions_ta,
                        obs_tas,
                        priors_ta,
                        obs,
                        prior,
                        dangling,
                        lengths, # Unused until implementing variable length
                        finished)
                else:
                    actions_ta, obs_tas, priors_ta, obs, prior, dangling, lengths, finished = loop_state
                    logits = cell_output + prior
                    next_cell_state = cell_state
                    emit_output = logits
                    action = tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32, seed=1)[:, 0]
                    # When implementing variable length:
                    # action = tf.where(
                    #     tf.logical_not(finished),
                    #     tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32)[:, 0],
                    #     tf.zeros(shape=[self.batch_size], dtype=tf.int32))
                    next_actions_ta = actions_ta.write(time - 1, action) # Write chosen actions
                    next_obs, next_prior, next_dangling = get_next_obs_prior_dangling(next_actions_ta, dangling)
                    next_input = get_input(next_obs)
                    next_obs_tas = ( # Write OLD observation
                        obs_tas[0].write(time - 1, obs[0]), # Action inputs
                        obs_tas[1].write(time - 1, obs[1]), # Parent inputs
                        obs_tas[2].write(time - 1, obs[2])) # Sibling inputs
                    next_priors_ta = priors_ta.write(time - 1, prior) # Write OLD prior
                    finished = next_finished = tf.logical_or(
                        finished,
                        time >= max_length)
                    # When implementing variable length:
                    # finished = next_finished = tf.logical_or(tf.logical_or(
                    #     finished, # Already finished
                    #     next_dangling == 0), # Currently, this will be 0 not just the first time, but also at max_length
                    #     time >= max_length)
                    next_lengths = tf.where(
                        finished, # Ever finished
                        lengths,
                        tf.tile(tf.expand_dims(time + 1, 0), [self.batch_size]))
                    next_loop_state = (next_actions_ta,
                                       next_obs_tas,
                                       next_priors_ta,
                                       next_obs,
                                       next_prior,
                                       next_dangling,
                                       next_lengths,
                                       next_finished)

                return (finished, next_input, next_cell_state, emit_output, next_loop_state)

            # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
            with tf.variable_scope('policy'):
                _, _, loop_state = tf.nn.raw_rnn(cell=cell, loop_fn=loop_fn)
                actions_ta, obs_tas, priors_ta, _, _, _, _, _ = loop_state

            self.actions = tf.transpose(actions_ta.stack(), perm=[1, 0]) # (?, max_length)
            self.obs = [tf.transpose(obs_ta.stack(), perm=[1, 0]) for obs_ta in obs_tas] # [(?, max_length)] * 3
            self.priors = tf.transpose(priors_ta.stack(), perm=[1, 0, 2]) # (?, max_length, n_choices)


        # Generates dictionary containing placeholders needed for a batch of sequences
        def make_batch_ph(name):
            with tf.name_scope(name):
                batch_ph = {
                    "actions" : tf.placeholder(tf.int32, [None, max_length]),
                    "obs" : (tf.placeholder(tf.int32, [None, max_length]),
                             tf.placeholder(tf.int32, [None, max_length]),
                             tf.placeholder(tf.int32, [None, max_length])),
                    "priors" : tf.placeholder(tf.float32, [None, max_length, n_choices]),
                    "lengths" : tf.placeholder(tf.int32, [None,]),
                    "rewards" : tf.placeholder(tf.float32, [None], name="r")
                }
                batch_ph = Batch(**batch_ph)

            return batch_ph

        def safe_cross_entropy(p, logq, axis=-1):
            safe_logq = tf.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
            return - tf.reduce_sum(p * safe_logq, axis)

        # Generates tensor for neglogp of a given batch
        def make_neglogp_and_entropy(B):
            with tf.variable_scope('policy', reuse=True):
                logits, _ = tf.nn.dynamic_rnn(cell=cell,
                                              inputs=get_input(B.obs),
                                              sequence_length=B.lengths, # Backpropagates only through sequence length
                                              dtype=tf.float32)
            logits += B.priors
            probs = tf.nn.softmax(logits)
            logprobs = tf.nn.log_softmax(logits)

            # Generate mask from sequence lengths
            # NOTE: Using this mask for neglogp and entropy actually does NOT
            # affect training because gradients are zero outside the lengths.
            # However, the mask makes tensorflow summaries accurate.
            mask = tf.sequence_mask(B.lengths, maxlen=max_length, dtype=tf.float32)

            # Negative log probabilities of sequences
            actions_one_hot = tf.one_hot(B.actions, depth=n_choices, axis=-1, dtype=tf.float32)
            neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2) # Sum over action dim
            neglogp = tf.reduce_sum(neglogp_per_step * mask, axis=1) # Sum over time dim

            # NOTE 1: The above implementation is the same as the one below:
            # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
            # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
            # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
            #   Exactly equivalent when removing priors.
            #   Equivalent up to precision when including clipped prior.
            #   Crashes when prior is not clipped due to multiplying zero by -inf.
            # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
            # neglogp_per_step = tf.reduce_sum(neglogp_per_step, axis=2)
            # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
            
            entropy_per_step = safe_cross_entropy(probs, logprobs, axis=2) # Sum over action dim -> (batch_size, max_length)
            entropy = tf.reduce_sum(entropy_per_step * mask, axis=1) # Sum over time dim -> (batch_size, )   
                    
            return neglogp, entropy


        # On policy batch
        self.sampled_batch_ph = make_batch_ph("sampled_batch")

        # Memory batch
        self.memory_batch_ph = make_batch_ph("memory_batch")
        memory_neglogp, _ = make_neglogp_and_entropy(self.memory_batch_ph)
        self.memory_probs = tf.exp(-memory_neglogp)
        self.memory_logps = -memory_neglogp

        # PQT batch
        if pqt:
            self.pqt_batch_ph = make_batch_ph("pqt_batch")

        # Setup losses
        with tf.name_scope("losses"):

            neglogp, entropy = make_neglogp_and_entropy(self.sampled_batch_ph)
            r = self.sampled_batch_ph.rewards

            # Entropy loss
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy, name="entropy_loss")
            loss = entropy_loss

            # PPO loss
            if ppo:
                assert not pqt, "PPO is not compatible with PQT"

                self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="old_neglogp")
                ratio = tf.exp(self.old_neglogp_ph - neglogp)
                clipped_ratio = tf.clip_by_value(ratio, 1. - ppo_clip_ratio, 1. + ppo_clip_ratio)
                ppo_loss = -tf.reduce_mean(tf.minimum(ratio * (r - self.baseline), clipped_ratio * (r - self.baseline)))
                loss += ppo_loss

                # Define PPO diagnostics
                clipped = tf.logical_or(ratio < (1. - ppo_clip_ratio), ratio > 1. + ppo_clip_ratio)
                self.clip_fraction = tf.reduce_mean(tf.cast(clipped, tf.float32))
                self.sample_kl = tf.reduce_mean(neglogp - self.old_neglogp_ph)

            # Policy gradient loss
            else:
                if not pqt or (pqt and pqt_use_pg):
                    pg_loss = tf.reduce_mean((r - self.baseline) * neglogp, name="pg_loss")                    
                    loss += pg_loss

            # Priority queue training loss
            if pqt:
                pqt_neglogp, _ = make_neglogp_and_entropy(self.pqt_batch_ph)
                pqt_loss = pqt_weight * tf.reduce_mean(pqt_neglogp, name="pqt_loss")
                loss += pqt_loss

            self.loss = loss

        def make_optimizer(name, learning_rate):
            if name == "adam":
                return tf.train.AdamOptimizer(learning_rate=learning_rate)
            if name == "rmsprop":
                return tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99)
            if name == "sgd":
                return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            raise ValueError("Did not recognize optimizer '{}'".format(name))

        # Create training op
        optimizer = make_optimizer(name=optimizer, learning_rate=learning_rate)
        with tf.name_scope("train"):
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            # The two lines above are equivalent to:
            # self.train_op = optimizer.minimize(self.loss)
        with tf.name_scope("grad_norm"):
            self.grads, _ = list(zip(*self.grads_and_vars))
            self.norms = tf.global_norm(self.grads)

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
                tf.summary.scalar("reward", tf.reduce_mean(r))
                tf.summary.scalar("baseline", self.baseline)
                tf.summary.histogram("reward", r)
                tf.summary.histogram("length", self.sampled_batch_ph.lengths)
                for g, v in self.grads_and_vars:
                    tf.summary.histogram(v.name, v)
                    tf.summary.scalar(v.name + '_norm', tf.norm(v))
                    tf.summary.histogram(v.name + '_grad', g)
                    tf.summary.scalar(v.name + '_grad_norm', tf.norm(g))
                tf.summary.scalar('gradient norm', self.norms)
                self.summaries = tf.summary.merge_all()

    def sample(self, n):
        """Sample batch of n expressions"""
        
        feed_dict = {self.batch_size : n}

        actions, obs, priors = self.sess.run([self.actions, self.obs, self.priors], feed_dict=feed_dict)

        return actions, obs, priors


    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""

        feed_dict = {
            self.memory_batch_ph : memory_batch
        }

        if log:
            fetch = self.memory_logps
        else:
            fetch = self.memory_probs
        probs = self.sess.run([fetch], feed_dict=feed_dict)[0]
        return probs


    def train_step(self, b, sampled_batch, pqt_batch):
        """Computes loss, trains model, and returns summaries."""

        feed_dict = {
            self.baseline : b,
            self.sampled_batch_ph : sampled_batch
        }

        if self.pqt:
            feed_dict.update({
                self.pqt_batch_ph : pqt_batch
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
                    mb_feed_dict = {k : v[mb] for k, v in feed_dict.items() if k not in [self.baseline, self.batch_size]}
                    mb_feed_dict.update({
                        self.baseline : b,
                        self.batch_size : len(mb)
                        })

                    _ = self.sess.run([self.train_op], feed_dict=mb_feed_dict)

        else:
            _ = self.sess.run([self.train_op], feed_dict=feed_dict)

        # Return summaries
        if self.summary:
            summaries = self.sess.run(self.summaries, feed_dict=feed_dict)
        else:
            summaries = None

        return summaries
