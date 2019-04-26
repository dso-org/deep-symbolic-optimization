from datetime import datetime

import tensorflow as tf
import numpy as np
from scipy.special import softmax
from sympy.parsing.sympy_parser import parse_expr

from dsr.program import Program
from dsr.expression import Dataset
import dsr.utils as U

tf.random.set_random_seed(0)

class Controller():
    def __init__(self, sess, num_units, library_size, max_length, learning_rate=0.001, batch_size=1, entropy_weight=0.0):

        self.sess = sess # TensorFlow session
        self.logits = [] # Outputs of the controller
        self.actions = [] # Actions sampled from the controller        
        self.actions_ph = [] # Placeholders for actions when computing probabilities and losses

        # Hyperparameters
        self.entropy_weight = entropy_weight

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
                action = tf.multinomial(logits, num_samples=1)
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
            self.logits_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, batch_size), name="logits_mask")

            neglogps = []
            entropies = []
            for i in range(max_length):

                action = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
                self.actions_ph.append(action)

                logits = self.logits[i]
                # labels = tf.squeeze(self.actions[i]) # Currently, actions must be fed in as placeholders...
                
                # Since labels are one-hot tensors, cross-entropy loss is equivalent to neglogp(action)
                neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=action)
                entropy = neglogp * tf.exp(-neglogp) # Entropy = neglogp * p = neglogp * exp(-neglogp)

                entropies.append(entropy)
                neglogps.append(neglogp)

            # Apply mask based on true length of sampled expression
            neglogps = tf.stack(neglogps) * self.logits_mask
            entropies = tf.stack(entropies) * self.logits_mask

            self.sample_neglogp = tf.reduce_sum(neglogps, axis=0) # Shape: (batch_size,)
            self.sample_entropy = tf.reduce_sum(entropies, axis=0) # Shape: (batch_size,)
            
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
        tf.summary.histogram("length", tf.reduce_sum(self.logits_mask, axis=0))
        self.summaries = tf.summary.merge_all()

        # Create training op
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)        


    def get_actions(self):

        logits, actions = self.sess.run([self.logits, self.actions])
        # actions = np.squeeze(np.stack(actions, axis=-1)) # Stack list and reshape to (batch_size, max_length)
        return logits, actions


    def train_step(self, r, b, actions, logits, logits_mask):

        feed_dict = {self.r : r,
                    self.baseline : b,
                    self.logits_mask : logits_mask}

        for i, action in enumerate(actions):
            feed_dict[self.actions_ph[i]] = np.squeeze(action)
        # for i, logit in enumerate(logits):
        #     feed_dict[self.logits_ph[i]] = np.squeeze(logit)

        loss, _, summaries = self.sess.run([self.loss, self.train_op, self.summaries], feed_dict=feed_dict)
        
        return loss, summaries


if __name__ == "__main__":

    # # Create ground-truth expression: sin(x^2) + cos(x*y)*sin(x^2)
    # ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x1,x1))))")
    # ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x1","x1"]) + [-1]*1, dtype=int).reshape(1, -1)
    # dataset = Dataset(ground_truth)

    # Create ground-truth expression: sin(x^2) + cos(x*y)*sin(y^2)
    ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x2,x2))))")
    ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x2","x2"]) + [-1]*1, dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)

    # Create ground-truth expression: sin(x^2) + cos(x*y) / sin(y^2)
    # ground_truth = Program(U.convert(["Add","sin","Mul","x1","x1","Div","cos","Mul","x1","x2","sin","Mul","x2","x2"]) + [-1]*1)
    # print(program.fraction(dataset.X, dataset.y))

    # Select hyperparameters
    num_units = 35
    library_size = U.n_choices
    max_length = 30
    batch_size = 1000

    # Start the TensorFlow session
    with tf.Session() as sess:

        # Create the controller
        controller = Controller(sess, num_units, library_size, max_length, batch_size=batch_size)

        # Create the summary writer
        logdir = "./summary/{}/".format(datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Initialize variables
        sess.run(tf.global_variables_initializer())        

        # Main training loop
        epochs = 1000
        best = -np.inf # Best reward
        b = None # Baseline used for control variates
        alpha = 0.1 # Coefficient used for EWMA
        for step in range(epochs):

            logits, actions = controller.get_actions() # Sample batch of expressions from controller
            # print("p0:", softmax(logits[0][0,:]))

            # unique_actions, counts = np.unique(np.squeeze(np.stack(actions, axis=-1))[:,:5], axis=0, return_counts=True)
            # print(unique_actions.shape[0])
            
            programs = [Program(a) for a in np.squeeze(np.stack(actions, axis=-1))] # Instantiate expressions
            r = np.array([p.fraction(dataset.X, dataset.y, alpha=0.25, epsilon=0.0) for p in programs]) # Compute reward
            b = np.mean(0) if b is None else alpha*np.mean(r) + (1 - alpha)*b # Compute baseline (EWMA of average reward)

            logits_mask = np.zeros_like(np.stack(np.squeeze(actions)), dtype=np.float32) # Shape: (max_length, batch_size)
            for i in range(batch_size):
                length = min(len(programs[i].program), max_length)
                logits_mask[:length, i] = 1.0

            loss, summaries = controller.train_step(r, b, actions, logits, logits_mask) # Train controller
            writer.add_summary(summaries, step)
            writer.flush()

            # print("Step: {}, Loss: {:.6f}, baseline: {:.6f}, r: {:.6f}".format(step, loss, b, np.mean(r)))
            if step > 0 and step % 10 == 0:
                print("Completed {} steps".format(step))

            if max(r) > best:
                index = np.argmax(r)
                best = r[index]
                print("New best expression: {} (reward = {})".format(programs[index], best))










