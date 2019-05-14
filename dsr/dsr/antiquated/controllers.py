import tensorflow as tf

import utils as U


class Controller():
    def __init__(self):
        self.logits = None

    def sample_trav(self):
        raise NotImplementedError

    def loglikelihood(self, traversal):
        raise NotImplementedError


class VectorController(Controller):
    """Controller is a fixed parameter vector with length equal to the library size"""

    def __init__(self):

        self.logits = tf.get_variable("logits",
                            shape=(U.n_choices),
                            trainable=True,
                            initializer=tf.zeros_initializer())

    def loglikelihood(self, trav):
        logp_all = tf.nn.log_softmax(self.logits)
        logp_trav = tf.reduce_sum(tf.one_hot(trav, depth=U.n_choices) * logp_all)
        return logp_trav


class MLPController(Controller):
    """Controller is a feedforward network that outputs a fixed paramter vector with length equal to the library size"""

    def __init__(self):

        self.inputs = tf.constant(1.0, dtype=tf.float32, shape=(1, U.n_choices))

        h1 = tf.contrib.layers.fully_connected(
            inputs=self.inputs,
            num_outputs=64,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.zeros_initializer(),
            biases_initializer=tf.zeros_initializer())

        self.logits = tf.contrib.layers.fully_connected(
            inputs=h1,
            num_outputs=U.n_choices,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer(),
            biases_initializer=tf.zeros_initializer())

    def loglikelihood(self, trav):
        logp_all = tf.nn.log_softmax(self.logits)
        logp_trav = tf.reduce_sum(tf.one_hot(trav, depth=U.n_choices) * logp_all)
        return logp_trav


class RNNController(Controller):
    """Controller is a recurrent network that outputs a parameter vector for each node in the expression tree"""

    def __init__(self):

        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, U.n_choices))

        lstm_cell = tf.contrib.rnn.LSTMCell(
            num_units=35)

        initial_state = lstm_cell.zero_state(None, dtype=tf.float32)
        probabilities = []

        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, self.inputs)

        self.logits = None


    def loglikelihood(self, trav):
        pass


    def sample_trav(self):
        trav = [-1] * U.MAX_SEQUENCE_LENGTH # -1 corresponds to empty choice (will not contribute to logp)
        count = 1
        for i in range(U.MAX_SEQUENCE_LENGTH):
            # logits, states = self.sess.run([self.logits, self.final_state])
            val, state, logp_val = ...





