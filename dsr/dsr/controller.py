from datetime import datetime
import itertools

import tensorflow as tf
import numpy as np
from scipy.special import softmax
from sympy.parsing.sympy_parser import parse_expr

from dsr.program import Program
from dsr.expression import Dataset
import dsr.utils as U

tf.random.set_random_seed(0)

class Controller():
    def __init__(self, sess, num_units, library_size, max_length, learning_rate=0.001, entropy_weight=0.0):

        self.sess = sess # TensorFlow session
        self.actions = [] # Actions sampled from the controller        
        self.logits = []        

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.max_length = max_length

        neglogps = []
        entropies = []

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
            input_dims = tf.stack([self.batch_size, 1, library_size])
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
                logits = tf.layers.dense(ouputs[:, -1, :], units=library_size)
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
                cell_input = tf.one_hot(tf.reshape(self.actions_ph[i], (self.batch_size, 1)), depth=library_size, name="cell_input_{}".format(i))
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
        """Samples n actions from the controller"""

        actions = []

        feed_dict = {self.batch_size : n}
        for i in range(self.max_length):
            action = self.sess.run(self.actions[i], feed_dict=feed_dict)
            actions.append(action)
            feed_dict[self.actions_ph[i]] = action

        return actions


    def neglogp(self, actions, actions_mask):
        """Returns neglogp of actions"""

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


if __name__ == "__main__":

    # Select hyperparameters
    num_units = 35
    library_size = U.n_choices
    max_length = 30
    batch_size = 1000

    # # Create ground-truth expression: sin(x^2) + cos(x*y)*sin(x^2)
    # ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x1,x1))))")
    # ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x1","x1"]) + [-1]*1, dtype=int).reshape(1, -1)
    # dataset = Dataset(ground_truth)

    # Create ground-truth expression: sin(x^2) + cos(x*y)*sin(y^2)
    ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x2,x2))))")
    ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x2","x2"]) + [-1]*1, dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)

    # Create ground-truth expression: cos(x*y) + sin(y^2)
    ground_truth = parse_expr("Add(cos(Mul(x1,x2)),sin(Mul(x2,x2)))")
    ground_truth_trav = np.array(U.convert(["Add","cos","Mul","x1","x2","sin","Mul","x2","x2"]), dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)

    # Create ground-truth expression: x1 + x2 + x3 + x4 + x5
    ground_truth = parse_expr("Add(Add(Add(Add(x1,x2),x3),x4),x5)")
    ground_truth_trav = np.array(U.convert(["Add","Add","Add","Add","x1","x2","x3","x4","x5"]), dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)

    # ground_truth_program = Program(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x2","x2"]) + [-1]*1)
    # ground_truth_actions = U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x2","x2"]) + [0]*(max_length - 14)
    # ground_truth_actions = np.vstack([np.array(ground_truth_actions)]*batch_size) # (batch_size, max_length)
    # ground_truth_actions = [ground_truth_actions[:, i] for i in range(ground_truth_actions.shape[1])]
    # ground_truth_actions_mask = np.zeros_like(np.stack(np.squeeze(ground_truth_actions)), dtype=np.float32)
    # ground_truth_actions_mask[:14, :] = 1.0

    # Start the TensorFlow session
    with tf.Session() as sess:

        # Create the controller
        controller = Controller(sess, num_units, library_size, max_length)

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

            actions = controller.sample(batch_size) # Sample batch of expressions from controller
            actions = np.squeeze(np.stack(actions, axis=-1)) # Shape (batch_size, max_length)

            # unique_actions, counts = np.unique(np.squeeze(np.stack(actions, axis=-1))[:,:5], axis=0, return_counts=True)
            # print(unique_actions.shape[0])
            
            programs = [Program(a) for a in actions] # Instantiate expressions
            r = np.array([p.fraction(dataset.X, dataset.y, alpha=0.25, epsilon=0.0) for p in programs]) # Compute reward

            # Heuristic: Only train on top epsilon percentile of sampled expressions
            epsilon = 99
            cutoff = r >= np.percentile(r, epsilon)
            actions = actions[cutoff, :]
            programs = list(itertools.compress(programs, cutoff))
            r = r[cutoff]

            b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b # Compute baseline (EWMA of average reward)

            # Compute actions mask
            actions_mask = np.zeros_like(actions.T, dtype=np.float32) # Shape: (max_length, batch_size)
            for i,p in enumerate(programs):
                length = min(len(p.program), max_length)
                actions_mask[:length, i] = 1.0

            loss, summaries = controller.train_step(r, b, actions, actions_mask) # Train controller
            writer.add_summary(summaries, step)
            writer.flush()

            # print("Step: {}, Loss: {:.6f}, baseline: {:.6f}, r: {:.6f}".format(step, loss, b, np.mean(r)))
            if step > 0 and step % 10 == 0:
                print("Completed {} steps".format(step))
                # print("Neglogp of ground truth action:", controller.neglogp(ground_truth_actions, ground_truth_actions_mask)[0])

            if max(r) > best:
                index = np.argmax(r)
                best = r[index]
                print("New best expression: {} (reward = {})".format(programs[index], best))