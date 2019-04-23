import tensorflow as tf
import numpy as np
from scipy.special import softmax
from sympy.parsing.sympy_parser import parse_expr

from program import Program
from dsr.expression import Dataset
import dsr.utils as U

tf.random.set_random_seed(0)

class Controller():
    def __init__(self, sess, num_units, library_size, max_length, learning_rate=0.001, batch_size=1):

        self.sess = sess # TensorFlow session
        self.logits = [] # Outputs of the controller
        self.actions = [] # Actions sampled from the controller        
        self.actions_ph = [] # Placeholders for actions when computing probabilities and losses

        # Build controller RNN
        with tf.name_scope("controller"):
            
            cell = tf.nn.rnn_cell.LSTMCell(num_units)
            cell_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            cell_input = tf.constant(1.0, dtype=tf.float32, shape=(batch_size, 1, library_size)) # First input fed to controller

            #####
            # TBD: Create embedding layer
            #####

            for i in range(max_length):
                ouputs, final_state = tf.nn.dynamic_rnn(cell,
                                                        cell_input,
                                                        initial_state=cell_state,
                                                        dtype=tf.float32)

                # Outputs correspond to logits of library
                logits = tf.layers.dense(ouputs[:, -1, :], units=library_size)

                # Sample from the library
                action = tf.multinomial(tf.nn.softmax(logits), num_samples=1)                
                cell_input = tf.one_hot(action, depth=library_size)

                cell_state = final_state

                self.actions.append(action) # Sampled values
                self.logits.append(logits) # Corresponding logits

            # print(self.actions)
            # print(self.logits)

        # Setup losses
        with tf.name_scope("losses"):

            self.r = tf.placeholder(dtype=tf.float32, shape=(batch_size,), name="r")
            self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")

            cross_entropy_loss = 0
            for i in range(max_length):
                logits = self.logits[i]
                action = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
                self.actions_ph.append(action)
                # labels = tf.squeeze(self.actions[i])

                # Since labels are one-hot tensors, cross-entropy loss is equivalent to neglogp(action)
                cross_entropy_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                    labels=action)
            
            # Policy gradient loss is neglogp(actions) scaled by reward
            policy_gradient_loss = tf.reduce_mean((self.r - self.baseline) * cross_entropy_loss, name="policy_gradient_loss")

        self.loss = policy_gradient_loss # May add additional terms later

        # Create training op
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)


    def get_actions(self):

        logits, actions = self.sess.run([self.logits, self.actions])
        # actions = np.squeeze(np.stack(actions, axis=-1)) # Stack list and reshape to (batch_size, max_length)
        return logits, actions


    def train_step(self, r, b, actions):

        feed_dict = {self.r : r,
                    self.baseline : b}

        for i, action in enumerate(actions):
            feed_dict[self.actions_ph[i]] = np.squeeze(action)

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss


if __name__ == "__main__":

    # Create ground-truth expression: sin(x^2) + cos(x*y)*sin(x^2)
    ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x1,x1))))")
    ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x1","x1"]) + [-1]*1, dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)

    # Select hyperparameters
    num_units = 35
    library_size = U.n_choices
    max_length = 30
    batch_size = 1000

    # Start the TensorFlow session
    sess = tf.Session()

    # Create the controller
    controller = Controller(sess, num_units, library_size, max_length, batch_size=batch_size)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Main training loop
    epochs = 1000
    best = -np.inf # Best reward
    b = None # Baseline used for control variates
    alpha = 0.001 # Coefficient used for EWMA
    for epoch in range(epochs):

        logits, actions = controller.get_actions() # Sample batch of expressions from controller

        # print("p0:", softmax(logits[0][0,:]))
        programs = [Program(a) for a in np.squeeze(np.stack(actions, axis=-1))] # Instantiate expressions
        r = np.array([p.fraction(dataset.X, dataset.y) for p in programs]) # Compute reward
        b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b # Compute baseline (EWMA of average reward)
        loss = controller.train_step(r, b, actions) # Train controller
        print("Loss:", loss)

        if max(r) > best:
            index = np.argmax(r)
            best = r[index]
            print("New best expression: {} (reward = {})".format(programs[index], best))










