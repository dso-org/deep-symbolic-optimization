"""Controller used to generate distribution over hierarchical, variable-length objects."""
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch

from dso.policy import Policy


class LinearWrapper(tf.keras.layers.Layer):
    """Layer wrapper that adds a linear (Dense) layer to an RNN cell's output.

    This is a TF2/Keras replacement for the TF1 LayerRNNCell wrapper.
    """

    def __init__(self, cell, output_size, **kwargs):
        super(LinearWrapper, self).__init__(**kwargs)
        self.cell = cell
        self._output_size = output_size
        self.dense = tf.keras.layers.Dense(output_size, name="output_dense")

    def call(self, inputs, states, training=None):
        outputs, new_states = self.cell(inputs, states, training=training)
        logits = self.dense(outputs)
        return logits, new_states

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # TF 2.17+ StackedRNNCells only accepts batch_size
        return self.cell.get_initial_state(batch_size=batch_size)


def safe_cross_entropy(p, logq, axis=-1):
    """Compute p * logq safely, by susbstituting
    logq[index] = 1 for index such that p[index] == 0
    """
    # Put 1 where p == 0. In the case, q =p, logq = -inf and this
    # might procude numerical errors below
    safe_logq = tf.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
    # Safely compute the product
    return - tf.reduce_sum(p * safe_logq, axis)


class RNNPolicy(Policy):
    """Recurrent neural network (RNN) policy used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees.

    Parameters
    ----------
    action_prob_lowerbound: float
        Lower bound on probability of each action.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    max_attempts_at_novel_batch: int
        maximum number of repetitions of sampling to get b new samples
        during a call of policy.sample(b)

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer.

    sample_novel_batch: bool
        if True, then a call to policy.sample(b) attempts to produce b samples
        that are not contained in the cache

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.

    """
    def __init__(self, prior, state_manager,
                 debug = 0,
                 max_length = 30,
                 action_prob_lowerbound = 0.0,
                 max_attempts_at_novel_batch = 10,
                 sample_novel_batch = False,
                 # RNN cell hyperparameters
                 cell ='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros'):
        super().__init__(prior, state_manager, debug, max_length)

        assert 0 <= action_prob_lowerbound  and action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        # len(tokens) in library
        self.n_choices = Program.library.L

        # setup model
        self._setup_tf_model(cell, num_layers, num_units, initializer)

        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch

    def _setup_tf_model(
            self,
            cell ='lstm',
            num_layers=1,
            num_units=32,
            initializer='zeros'):

        # Defined in super class
        # This can be susbtituted below
        n_choices = self.n_choices
        prior = self.prior
        state_manager = self.state_manager
        max_length = self.max_length

        # Build RNN policy
        def make_initializer(name):
            if name == "zeros":
                return tf.initializers.Zeros()
            if name == "var_scale":
                return tf.keras.initializers.VarianceScaling(
                        scale=0.5, mode='fan_avg', distribution='uniform', seed=0)
            raise ValueError("Did not recognize initializer '{}'".format(name))

        def make_cell(name, num_units, initializer):
            if name == 'lstm':
                return tf.keras.layers.LSTMCell(num_units, kernel_initializer=initializer,
                                                 recurrent_initializer=initializer)
            if name == 'gru':
                return tf.keras.layers.GRUCell(num_units, kernel_initializer=initializer,
                                                bias_initializer=initializer,
                                                recurrent_initializer=initializer)
            raise ValueError("Did not recognize cell type '{}'".format(name))

        # Create recurrent cell
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers
        init = make_initializer(initializer)
        cells = [make_cell(cell, n, initializer=init) for n in num_units]
        if len(cells) > 1:
            stacked_cell = tf.keras.layers.StackedRNNCells(cells)
        else:
            stacked_cell = cells[0]
        self.cell = LinearWrapper(cell=stacked_cell, output_size=n_choices)

        # Store these for use in sampling and training
        self.task = Program.task
        state_manager.setup_manager(self)
        self._initial_obs = self.task.reset_task(prior)
        self._initial_prior = tf.constant(self.prior.initial_prior(), dtype=tf.float32)

        # For RNN layer used in make_neglogp_and_entropy
        self.rnn_layer = tf.keras.layers.RNN(
            self.cell,
            return_sequences=True,
            return_state=False
        )

    def _sample_step(self, batch_size):
        """Sample a batch of sequences using eager execution.

        This is a TF2 replacement for tf.nn.raw_rnn.
        """
        n_choices = self.n_choices
        max_length = self.max_length
        task = self.task
        state_manager = self.state_manager

        # Initialize lists for storing results
        all_actions = []
        all_obs = []
        all_priors = []

        # Get initial states - _initial_obs could be numpy or tensor
        if hasattr(self._initial_obs, 'numpy'):
            initial_obs_np = self._initial_obs.numpy()
        else:
            initial_obs_np = np.array(self._initial_obs)
        initial_obs = np.broadcast_to(initial_obs_np, [batch_size, len(initial_obs_np)])
        initial_obs = state_manager.process_state(tf.constant(initial_obs, dtype=tf.float32)).numpy()

        if hasattr(self._initial_prior, 'numpy'):
            initial_prior_np = self._initial_prior.numpy()
        else:
            initial_prior_np = np.array(self._initial_prior)
        initial_prior = np.broadcast_to(initial_prior_np, [batch_size, n_choices])

        # Get initial cell state (TF 2.17+ only accepts batch_size)
        cell_state = self.cell.get_initial_state(batch_size=batch_size)

        obs = initial_obs
        prior_t = initial_prior
        finished = np.zeros([batch_size], dtype=bool)

        for time in range(max_length):
            if np.all(finished):
                # Pad remaining timesteps with zeros
                remaining = max_length - time
                all_actions.extend([np.zeros(batch_size, dtype=np.int32)] * remaining)
                all_obs.extend([np.zeros((batch_size, obs.shape[1]), dtype=np.float32)] * remaining)
                all_priors.extend([np.zeros((batch_size, n_choices), dtype=np.float32)] * remaining)
                break

            # Get input from state manager
            obs_tensor = tf.constant(obs, dtype=tf.float32)
            input_t = state_manager.get_tensor_input(obs_tensor)

            # Run cell
            cell_output, cell_state = self.cell(input_t, cell_state)

            # Apply action prob lower bound
            if self.action_prob_lowerbound != 0.0:
                cell_output = self.apply_action_prob_lowerbound(cell_output)

            logits = cell_output.numpy() + prior_t

            # Sample action
            action = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)[:, 0].numpy()

            # Store results
            all_actions.append(action)
            all_obs.append(obs.copy())
            all_priors.append(prior_t.copy())

            # Compute next obs and prior
            actions_so_far = np.stack(all_actions, axis=1)  # (batch, time+1)
            next_obs, next_prior, next_finished = task.get_next_obs(
                actions_so_far, obs, finished
            )

            obs = state_manager.process_state(tf.constant(next_obs, dtype=tf.float32)).numpy()
            prior_t = next_prior
            finished = next_finished | (time + 1 >= max_length)

        # Stack results
        actions = np.stack(all_actions, axis=1)  # (batch, max_length)
        obs_stacked = np.stack(all_obs, axis=2)  # (batch, obs_dim, max_length)
        priors = np.stack(all_priors, axis=1)  # (batch, max_length, n_choices)

        return actions, obs_stacked, priors

    def make_neglogp_and_entropy(self, B, entropy_gamma):
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.

        Returns
        -------
        neglogp, entropy :
            Tensorflow tensors
        """

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        entropy_gamma_decay = np.array([entropy_gamma**t for t in range(self.max_length)], dtype=np.float32)

        # Prepare inputs - transpose obs from (batch, obs_dim, time) to (batch, time, obs_dim)
        obs_transposed = tf.transpose(B.obs, perm=[0, 2, 1])
        inputs = self.state_manager.get_tensor_input(obs_transposed)

        # Run RNN with masking
        mask = tf.sequence_mask(B.lengths, maxlen=tf.shape(B.actions)[1])
        logits = self.rnn_layer(inputs, mask=mask)

        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits += B.priors
        probs = tf.nn.softmax(logits)
        logprobs = tf.nn.log_softmax(logits)
        B_max_length = tf.shape(B.actions)[1]  # Maximum sequence length for this Batch

        # Generate mask from sequence lengths
        mask_float = tf.sequence_mask(B.lengths, maxlen=B_max_length, dtype=tf.float32)

        # Negative log probabilities of sequences
        actions_one_hot = tf.one_hot(B.actions, depth=self.n_choices, axis=-1, dtype=tf.float32)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2)  # Sum over action dim
        neglogp = tf.reduce_sum(neglogp_per_step * mask_float, axis=1)  # Sum over time dim

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        sliced_entropy_gamma_decay = tf.slice(entropy_gamma_decay, [0], [B_max_length])
        entropy_gamma_decay_mask = sliced_entropy_gamma_decay * mask_float  # ->(batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, axis=2)  # Sum over action dim -> (batch_size, max_length)
        entropy = tf.reduce_sum(entropy_per_step * entropy_gamma_decay_mask, axis=1)  # Sum over time dim -> (batch_size, )

        return neglogp, entropy


    def sample(self, n : int):
        """Sample batch of n expressions

        Returns
        -------
        actions, obs, priors :
            Or a batch
        """
        if self.sample_novel_batch:
            actions, obs, priors = self.sample_novel(n)
        else:
            actions, obs, priors = self._sample_step(n)
            # Already numpy arrays

        return actions, obs, priors

    def sample_novel(self, n: int):
        """Sample a batch of n expressions not contained in cache.

        If unable to do so within self.max_attempts_at_novel_batch,
        then fills in the remaining slots with previously-seen samples.

        Parameters
        ----------
        n: int
            batch size

        Returns
        -------
        unique_a, unique_o, unique_p: np.ndarrays
        """
        n_novel = 0
        # Keep the samples that are produced by policy and already exist in cache,
        # so that DSO can train on everything
        old_a, old_o, old_p = [], [], []
        # Store the new samples separately for (expensive) reward evaluation
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            # [batch, time], [batch, obs_dim, time], [batch, time, n_choices]
            actions, obs, priors = self._sample_step(n)
            # Already numpy arrays
            n_attempts += 1
            new_indices = []  # indices of new and unique samples
            old_indices = []  # indices of samples already in cache
            for idx, a in enumerate(actions):
                tokens = _finish_tokens(a)
                key = tokens.tostring()
                if not key in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)
            # get all new actions, obs, priors in this group
            new_a.append(np.take(actions, new_indices, axis=0))
            new_o.append(np.take(obs, new_indices, axis=0))
            new_p.append(np.take(priors, new_indices, axis=0))
            old_a.append(np.take(actions, old_indices, axis=0))
            old_o.append(np.take(obs, old_indices, axis=0))
            old_p.append(np.take(priors, old_indices, axis=0))

        # number of slots in batch to be filled in by redundant samples
        n_remaining = n - n_novel

        # -------------------- combine all -------------------- #
        # Pad everything to max_length
        for tup, name in zip([(old_a, new_a), (old_o, new_o), (old_p, new_p)],
                                    ['action', 'obs', 'prior']):
            dim_length = 1 if name in ['action', 'prior'] else 2
            max_length = np.max([list_batch.shape[dim_length] for
                                 list_batch in tup[0] + tup[1]])
            # tup is a tuple of (old_?, new_?), each is a list of batches
            for list_batch in tup:
                for idx, batch in enumerate(list_batch):
                    n_pad = max_length - batch.shape[dim_length]
                    # Pad with 0 for everything because training step
                    # truncates based on each sample's own sequence length
                    # so the value does not matter
                    if name == 'action':
                        width = ((0,0),(0,n_pad))
                        vals = ((0,0),(0,0))
                    elif name == 'obs':
                        width = ((0,0),(0,0),(0,n_pad))
                        vals = ((0,0),(0,0),(0,0))
                    else:
                        width = ((0,0),(0,n_pad),(0,0))
                        vals = ((0,0),(0,0),(0,0))
                    list_batch[idx] = np.pad(
                        batch, pad_width=width, mode='constant',
                        constant_values=vals)

        old_a = np.concatenate(old_a)
        old_o = np.concatenate(old_o)
        old_p = np.concatenate(old_p)
        # If not enough novel samples, then fill in with redundancies
        new_a = np.concatenate(new_a + [old_a[:n_remaining]])
        new_o = np.concatenate(new_o + [old_o[:n_remaining]])
        new_p = np.concatenate(new_p + [old_p[:n_remaining]])

        # first entry serves to force object type, and also
        # indicates not to use it if zero
        self.extended_batch = np.array(
            [old_a.shape[0], old_a, old_o, old_p], dtype=object)
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""
        neglogp, _ = self.make_neglogp_and_entropy(memory_batch, None)

        if log:
            probs = -neglogp
        else:
            probs = tf.exp(-neglogp)

        return probs.numpy()

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to probabilities of each action.

        Parameters
        ----------
        logits: tf.Tensor where last dimension has size self.n_choices

        Returns
        -------
        logits_bounded: tf.Tensor
        """
        probs = tf.nn.softmax(logits, axis=-1)
        probs_bounded = ((1-self.action_prob_lowerbound)*probs +
                         self.action_prob_lowerbound/
                         float(self.n_choices))
        logits_bounded = tf.math.log(probs_bounded)

        return logits_bounded
