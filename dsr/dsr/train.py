import itertools

import tensorflow as tf
import numpy as np
from sympy.parsing.sympy_parser import parse_expr

from dsr import utils as U
from dsr.expression import Expression, Dataset


# Sample a pre-order tree traversal from distribution parameters p
def sample_trav(p):
    trav = [-1] * U.MAX_SEQUENCE_LENGTH # -1 corresponds to empty choice (will not contribute to logp)
    count = 1
    for i in range(U.MAX_SEQUENCE_LENGTH):

        try:
            val = np.random.choice(U.choices, p=p)
        except ValueError:
            print("ValueError:")
            print(i, trav, p)
            exit()
        trav[i] = val
        count += U.n_children[val] - 1

        # HACK: If you're going to reach the max length, adjust p to only choose leaf nodes
        if count >= U.MAX_SEQUENCE_LENGTH - i - 1:
            p[:-U.n_inputs] = 0
            p /= sum(p)
        elif count >= U.MAX_SEQUENCE_LENGTH - i - 2:
            p[:len(U.binary)] = 0
            p /= sum(p)

        if count == 0:
            break
        
    assert count == 0 or trav[-1] == -1, (count, U.convert(trav))
    return trav

def main():

    # Parameters of the distribution over expressions
    # For now, we will train these directly
    # In the future, these will be outputs of a neural network
    logits = tf.get_variable("logits",
                            shape=[U.n_choices],
                            trainable=True,
                            initializer=tf.zeros_initializer())

    # Define placeholders for traversal encoding, reward, and baseline
    trav_ph = tf.placeholder(dtype=tf.int32, shape=(None, U.MAX_SEQUENCE_LENGTH))
    r_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
    b_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Compute the log likelihood of a given per-order traversal
    def traversal_loglikelihood(trav):
        logp_all = tf.nn.log_softmax(logits)
        logp_trav = tf.reduce_sum(tf.one_hot(trav, depth=U.n_choices) * logp_all)
        return logp_trav

    # Define loss tensor and training operation
    logp_trav = traversal_loglikelihood(trav_ph)
    loss = -tf.reduce_mean(logp_trav * (r_ph - b_ph))
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Start tensorflow session and initialize variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create a dataset

    # sin(x) + sin(y^2)
    # ground_truth = parse_expr("Add(sin(x1),sin(Mul(x2,x2)))")
    # ground_truth_trav = np.array(U.convert(["Add","sin","x1","sin","Mul","x2","x2"]) + [-1]*8, dtype=int).reshape(1, -1)
    # dataset = Dataset(ground_truth)

    # sin(x^2) + cos(x*y)*sin(x^2)
    ground_truth = parse_expr("Add(sin(Mul(x1,x1)),Mul(cos(Mul(x1,x2)),sin(Mul(x1,x1))))")
    ground_truth_trav = np.array(U.convert(["Add","sin","Mul","x1","x1","Mul","cos","Mul","x1","x2","sin","Mul","x1","x1"]) + [-1]*1, dtype=int).reshape(1, -1)
    dataset = Dataset(ground_truth)


    # Main training loop
    epochs = 100
    trav_per_epoch = 1000
    best = -np.inf # Best reward
    b = None # Baseline used for control variates
    alpha = 0.2 # Coefficient used for EWMA
    for epoch in range(epochs):

        p = sess.run(tf.nn.softmax(logits)) # Current parameters
        travs = [sample_trav(p.copy()) for _ in range(trav_per_epoch)] # Sample traversals
        expressions = [Expression(traversal=trav) for trav in travs] # Instantiate expressions
        mse = np.array([expr.loss(dataset.X, dataset.y) for expr in expressions]) # Compute regression loss
        r = -mse # Compute reward (negative regression loss)
        b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b # Compute baseline (EWMA of average reward)
        print("MSE: {:.3f} +/- {:.3f}".format(np.mean(mse), np.std(mse)))
        print("MSE Q1/Q2/Q3: {:.3f}/{:.3f}/{:.3f}".format(np.percentile(mse, 25), np.percentile(mse, 50), np.percentile(mse, 75)))
        print("MSE Low/High: {:.3f}/{:.3f}".format(np.min(mse), np.max(mse)))
        print("Loglikelihood of correct answer: {:.3f}".format(sess.run(logp_trav, feed_dict={trav_ph : ground_truth_trav})))
        print("Parameter vector: {}".format(p))

        # Search for new best expression
        if max(r) > best:
            index = np.argmax(r)
            best = r[index]
            print("New best expression: {} (reward = {})".format(expressions[index], best))

        ###
        # logp = tf.log(tf.constant([1,1,2,1e-9,1,2], dtype=tf.float32)/7)
        # t = tf.constant(ground_truth_trav, dtype=tf.int32)
        # x = tf.reduce_sum(tf.one_hot(t, depth=U.n_choices) * logp)
        # print("Loglikelihood of hard-coded parameters", sess.run(x))
        # exit()
        # ###

        # HEURISTIC
        # Only train on >= epsilon percentile of sampled expressions
        epsilon = 90
        cutoff = r >= np.percentile(r, epsilon)
        travs = list(itertools.compress(travs, cutoff))
        r = r[cutoff]

        # Perform training and report loss
        L, _ = sess.run([loss, train_op], feed_dict={trav_ph : travs, r_ph : r, b_ph : b})
        print("Loss: {:.3f}".format(L))
        print("")


if __name__ == "__main__":
    main()
