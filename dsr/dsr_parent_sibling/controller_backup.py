import tensorflow as tf
import numpy as np

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

    n_choices : int
        Size of library of operators/terminals.

    max_length : int
        Maximum length of a sampled traversal.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.
    """

    def __init__(self, sess, library, num_units, n_choices, max_length, 
                 learning_rate=0.001, entropy_weight=0.0):

        self.sess = sess
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
        self.parents_ph = [] #[Soo]
        self.siblings_ph = []
        self.library = library

        self.r = tf.placeholder(dtype=tf.float32, shape=(None,), name="r")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
        self.actions_mask = tf.placeholder(dtype=tf.float32, shape=(max_length, None), name="actions_mask")

        # Build controller RNN
        with tf.name_scope("controller"):
            
            cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.zeros_initializer())
            cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            input_dims = tf.stack([self.batch_size, 1, n_choices*3]) 
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
                self.logits.append(logits)

                # Sample from the library
                action = tf.multinomial(logits, num_samples=1)
                action = tf.to_int32(action)
                action = tf.reshape(action, (self.batch_size,))
                self.actions.append(action)

                # Placeholder for selected actions : Soo,is shape =(batch_size, 1)?
                action_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                self.actions_ph.append(action_ph)

                #Soo: Placeholder for parent and sibling: as the form of index in dictionary
                parent_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
                self.parents_ph.append(parent_ph)
                sibling_ph = tf.placeholder(dtype=tf.int32, shape=(None,)) 
                self.siblings_ph.append(sibling_ph)

                # Update LSTM input and state with selected actions
                # output shape of cell_input[1-3] = [batch_size, 1, n_choices] = shape=(?, 1, 7)
                cell_input1 = tf.one_hot(tf.reshape(self.actions_ph[i], (self.batch_size, 1)), depth=n_choices, name="cell_input_{}".format(i)) #action
                cell_input2 = tf.one_hot(tf.reshape(self.parents_ph[i], (self.batch_size, 1)), depth=n_choices, name="cell_input_{}".format(i)) #parent
                cell_input3 = tf.one_hot(tf.reshape(self.siblings_ph[i], (self.batch_size, 1)), depth=n_choices, name="cell_input_{}".format(i)) #sibling
                
                #Soo: concatenate to cell_input
                # Somehow, the size of input dim of RNN is fixed to 3 [batch_size, time, size_of_input]
                cell_input = tf.concat([cell_input1, cell_input2, cell_input3], 2) #(?,1,21)
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

    # [Soo]: We calculate parent and sibling here simultaneously when we sample the action.
    # Our RNN is trained with in-situ fasion as we sample out action and feed them together with parent-sibling pair
    def sample(self, n):
        """Sample batch of n expressions
           actions:  [self.max_length,self.batch_size] """

        actions = []
        feed_dict = {self.batch_size : n}
        for i in range(self.max_length):
            action = self.sess.run(self.actions[i], feed_dict=feed_dict) #(1000,)
            actions.append(action) #(length, 1000)
            feed_dict[self.actions_ph[i]] = action #feed_forward
            #[Soo]
            parents = np.empty(np.shape(action))
            siblings = np.empty(np.shape(action))
            for j in range(n):
                action_list_so_far = np.asarray(actions)[:,j]
                parent, sibling = self.parent_sibling(action_list_so_far, self.library)
                parents[j]=parent
                siblings[j]= sibling
            feed_dict[self.parents_ph[i]] = parents
            feed_dict[self.siblings_ph[i]] = siblings
        return actions
   
    def neglogp(self, actions, actions_mask):
        """Returns neglogp of batch of expressions"""

        feed_dict = {self.actions_ph[i] : a for i,a in enumerate(actions.T)}
        feed_dict[self.actions_mask] = actions_mask
        feed_dict[self.batch_size] = actions.shape[0]

        return self.sess.run(self.sample_neglogp, feed_dict=feed_dict)
   
    
    def parent_sibling(self, array_of_sequence, library):
        """
          input: array_of_sequence: index of sampled actions
                                   shape: (length_of_sampled_sequence)
                                   ex: [1,2,3]
                 library: ['add', 'mul', 'sin', 'cos', 'const', 'x0', 'x1'] --> can be different (defined in config.json) 
          output: parent:index of parent 
                         shape: (1)   
                  sibling:index of sibling
                         shape: (1)
        """
        uniary=["sin","cos","tan"] 
        binary=["add","mul"]
        operand=["x0","x1","x2","const"]

        #Default if len(array_of_sequence) <1:
        parent = 0
        sibling = 0
        if library[array_of_sequence[-1]] in uniary:
            parent = array_of_sequence[-1]
            sibling = 0
        elif  library[array_of_sequence[-1]] in binary:
            parent = array_of_sequence[-1]
            sibling = 0
        elif library[array_of_sequence[-1]] in operand:
            sum_of_operand = 0
            for i in range(len(array_of_sequence)):
                if library[array_of_sequence[len(array_of_sequence)-i-1]] in operand: #read from backward
                    sum_of_operand += 1
                elif library[array_of_sequence[len(array_of_sequence)-i-1]] in binary:
                    sum_of_operand -= 1
                if sum_of_operand == 0:
                    parent = array_of_sequence[len(array_of_sequence)-i-1]
                    sibling= array_of_sequence[len(array_of_sequence)-i]
                    break
        return parent, sibling




    def train_step(self, r, b, actions, actions_mask):
        """Computes loss, applies gradients, and computes summaries
           [Soo]: This is the train_step function when we want to feed [parent-sibling] pair together with action in training stage.
           I don't know why do we need to feed_dict for parent and sibling when we train (as parent and sibling is not making effect to loss)
           parent-sibling pair are alredy feeded to the network at sample funciton above."""
        feed_dict = {self.r : r,
                    self.baseline : b,
                    self.actions_mask : actions_mask,
                    self.batch_size : actions.shape[0]}
        #Soo: feeding parent and sibling index as the additional inputs
        action_list_so_far = [[] for i in range(actions.shape[0])]
        parent_list = [[] for i in range(actions.shape[0])]
        sibling_list = [[] for i in range(actions.shape[0])]
        for i, action in enumerate(actions.T):
            print(np.shape(action))
            feed_dict[self.actions_ph[i]] = action 
            for j in range(len(action)):
                action_list_so_far[j].append(action[j]) #input for parent_sibling
                parent, sibling = self.parent_sibling(action_list_so_far[j], self.library)
                parent_list[j].append(parent)
                sibling_list[j].append(sibling)
            #np.shape(parent_list), np.shape(sibling_list)) =(100, 30) (100, 30) : (batch_size, max_length_of_expression)
            feed_dict[self.parents_ph[i]] = parent_list
            feed_dict[self.siblings_ph[i]] = sibling_list
        loss, _, summaries = self.sess.run([self.loss, self.train_op, self.summaries], feed_dict=feed_dict)   
        return loss, summaries
