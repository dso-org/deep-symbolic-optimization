import os
import tensorflow as tf
import numpy as np
from numba import jit, prange

from dsr.program import Program
from dsr.fileio import FileIO


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
    """

    def __init__(self, name, sess, num_units, max_length, learning_rate=0.001,
                 entropy_weight=0.0, observe_action=True, observe_parent=True,
                 observe_sibling=True, summary=True, constrain_const=True,
                 constrain_trig=True,output={}):

        self.sess = sess
        self.actions = [] # Actions sampled from the controller
        self.logits = []
        self.summary = summary

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.max_length = max_length
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.constrain_const = constrain_const
        self.constrain_trig = constrain_trig
        neglogps = []
        entropies = []

        n_choices = Program.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.actions_ph = []
        self.parents_ph = []
        self.siblings_ph = []
        self.priors_ph = []

        assert observe_action + observe_parent + observe_sibling > 0, "Must include at least one observation."
        self.compute_parents_siblings = any([self.observe_parent,
                                             self.observe_sibling,
                                             self.constrain_const])
        self.compute_prior = any([self.constrain_const,
                                  self.constrain_trig])

        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        self.actions_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, None), name="actions_mask")

#--- save run output
        self.output = None
        if len(output)>0:     
           outname = name+'_dsr.csv'
           if "file" in output:
              outname = output["file"]
           outname = os.path.join(output["dir"],outname)
           self.output = FileIO(outname,output["fields"])

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

            # Define prior on logits; currently only used to apply hard constraints
            # First node must be nonterminal
            arities = np.array([Program.arities[i] for i in range(n_choices)])
            prior = np.zeros(len(arities), dtype=np.float32)
            prior[arities == 0] = -np.inf
            prior = tf.constant(prior, dtype=tf.float32)

            for i in range(max_length):
                ouputs, final_state = tf.nn.dynamic_rnn(cell,
                                                        cell_input,
                                                        initial_state=cell_state,
                                                        dtype=tf.float32)

                # Outputs correspond to logits of library
                logits = tf.layers.dense(ouputs[:, -1, :], units=n_choices)
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
            
            # Policy gradient loss is neglogp(actions) scaled by reward
            #[SOO: to do: change as ppo]
            policy_gradient_loss = tf.reduce_mean((self.r - self.baseline) * self.sample_neglogp, name="policy_gradient_loss")

            # Entropy loss is negative entropy, since entropy provides a bonus
            entropy_loss = -self.entropy_weight*tf.reduce_mean(self.sample_entropy, name="entropy_loss")

            self.loss = policy_gradient_loss + entropy_loss # May add additional terms later

        # Create summaries
        if self.summary:
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

            if self.compute_parents_siblings or self.constrain_trig:
                tokens = np.stack(actions).T

            # Compute parents and siblings
            if self.compute_parents_siblings:
                tokens = np.stack(actions).T # Shape: (n, i)
                parents, siblings = parents_siblings(tokens, Program.arities_numba)
                if self.observe_parent:
                    feed_dict[self.parents_ph[i]] = parents # Shape: (n,)
                if self.observe_sibling:
                    feed_dict[self.siblings_ph[i]] = siblings # Shape: (n,)

            # Compute prior
            if self.compute_prior:
                prior = np.zeros((n, Program.L), dtype=np.float32)
                if self.constrain_const:
                    constraints = np.isin(parents, Program.unary_tokens) # Unary parent (or unary action)
                    constraints += siblings == Program.const_token # Constant sibling
                    prior += make_prior(constraints, [Program.const_token], Program.L)
                if self.constrain_trig:
                    constraints = trig_ancestors(tokens, Program.arities_numba, Program.trig_tokens)
                    prior += make_prior(constraints, Program.trig_tokens, Program.L)
                feed_dict[self.priors_ph[i]] = prior

        actions = np.stack(actions).T # Shape: (n, max_length)
        return actions


    def neglogp(self, actions, actions_mask):
        """Returns neglogp of batch of expressions"""
        self.output = None
        if len(output)>0:     
           outname = name+'_dsr.csv'
           if "file" in output:
              outname = output["file"]
           outname = os.path.join(output["dir"],outname)
           self.output = FileIO(outname,output["fields"])
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

            if self.compute_parents_siblings or self.constrain_trig:
                tokens = np.stack(all_actions).T

            # TBD: Why does parents_siblings() have to be recalculated? It's not a function of the loss...
            if self.compute_parents_siblings:
                parents, siblings = parents_siblings(tokens, Program.arities_numba)
                if self.observe_parent:
                    feed_dict[self.parents_ph[i]] = parents
                if self.observe_sibling:
                    feed_dict[self.siblings_ph[i]] = siblings

                # Compute prior
            if self.compute_prior:
                prior = np.zeros((len(r), Program.L), dtype=np.float32)
                if self.constrain_const:                    
                    constraints = np.isin(parents, Program.unary_tokens) # Unary parent (or unary action)
                    constraints += siblings == Program.const_token # Constant sibling
                    prior += make_prior(constraints, [Program.const_token], Program.L)
                if self.constrain_trig:
                    constraints = trig_ancestors(tokens, Program.arities_numba, Program.trig_tokens)
                    prior += make_prior(constraints, Program.trig_tokens, Program.L)
                feed_dict[self.priors_ph[i]] = prior

#--- changed loss to self.calc_loss because the output is configured by nameing the fields, which must have unique names 
        ops = [self.loss, self.train_op]
        if self.summary:
            self.calc_loss, _, summaries = self.sess.run(ops + [self.summaries], feed_dict=feed_dict)
            return self.calc_loss, summaries
        else:
            self.calc_loss, _ = self.sess.run(ops, feed_dict=feed_dict)
            return self.calc_loss, None

    def savestepinfo(self, sinfo):

        if self.output is None:
           return
        dt = {}
        for p in self.output.header:
            if p in sinfo:
               dt[p] = sinfo[p]
            else:
               dt[p] = getattr(self,p)
        
        self.output.update(dt)



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
def parents_siblings(tokens, arities):
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

    Returns
    _______

    parents : np.ndarray, shape=(N,), dtype=np.int32
        Parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """

    N, L = tokens.shape
    parents = np.full(shape=(N,), fill_value=-1, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=-1, dtype=np.int32)
    # Parallelized loop over action sequences
    for r in prange(N):
        arity = arities[tokens[r, -1]]
        if arity > 0: # Parent is the previous element; no sibling
            parents[r] = tokens[r, -1]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-c-1, sibling is the next
                parents[r] = tokens[r, L - c - 1]
                siblings[r] = tokens[r, L - c]
                break
    return parents, siblings

