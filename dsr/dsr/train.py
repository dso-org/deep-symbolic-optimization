import os
import json
from itertools import compress
from datetime import datetime
from textwrap import indent

import tensorflow as tf
import numpy as np

from dsr.controller import Controller
from dsr.program import Program
from dsr.dataset import Dataset


# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Set TensorFlow seed
tf.random.set_random_seed(0)


def learn(
    sess,               # TensorFlow Session object
    controller,         # Controller object
    X, y,               # X and y of dataset
    logdir=".",         # Name of log directory
    n_epochs=1000,      # Number of epochs
    batch_size=1000,    # Number of samples per epoch
    reward="neg_mse",   # Reward function names
    reward_params=None, # Reward function parameters
    alpha=0.1,          # Coefficient of exponentially-weighted moving average of baseline
    epsilon=0.01):      # Fraction of top expressions used for training    

    # Create the summary writer
    logdir = os.path.join("log", logdir)
    os.makedirs(logdir, exist_ok=True)
    logdir = "./summary/{}/".format(datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    writer = tf.summary.FileWriter(logdir, sess.graph)

    # Set the reward function
    reward_params = reward_params if reward_params is not None else []
    Program.set_reward_function(reward, *reward_params)

    # Initialize compute graph
    sess.run(tf.global_variables_initializer())        

    # Main training loop
    best = -np.inf # Best reward
    b = None # Baseline used for control variates
    for step in range(n_epochs):

        actions = controller.sample(batch_size) # Sample batch of expressions from controller
        actions = np.squeeze(np.stack(actions, axis=-1)) # Shape (batch_size, max_length)

        # unique_actions, counts = np.unique(np.squeeze(np.stack(actions, axis=-1))[:,:5], axis=0, return_counts=True)
        # print(unique_actions.shape[0])
        
        programs = [Program(a) for a in actions] # Instantiate expressions
        r = np.array([p.reward(X, y) for p in programs]) # Compute reward

        # Heuristic: Only train on top epsilon fraction of sampled expressions
        if epsilon is not None and epsilon < 1.0:
            cutoff = r >= np.percentile(r, 100 - int(100*epsilon))
            actions = actions[cutoff, :]
            programs = list(compress(programs, cutoff))
            r = r[cutoff]

        b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b # Compute baseline (EWMA of average reward)

        # Compute actions mask
        actions_mask = np.zeros_like(actions.T, dtype=np.float32) # Shape: (max_length, batch_size)
        for i,p in enumerate(programs):
            length = min(len(p.program), controller.max_length)
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
            print("\nNew best expression:")
            print("\treward = {}".format(best))
            print("\t{}".format(programs[index]))
            print("{}\n".format(indent(programs[index].pretty(), '\t')))


def main():

    # Load the config file
    config_filename = 'config.json'
    with open(config_filename, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]          # Problem specification parameters
    config_training = config["training"]        # Training hyperparameters
    config_controller = config["controller"]    # Controller hyperparameters

    # Create the dataset
    dataset = Dataset(**config_dataset)
    X, y = dataset.X_train, dataset.y_train
    print("Ground truth expression:\n{}".format(indent(dataset.pretty(), '\t')))

    # Define the library
    Program.set_library(config_dataset["operators"], X.shape[1])
    n_choices = len(Program.library)
    
    with tf.Session() as sess:

        # Instantiate the controller
        controller = Controller(sess, n_choices=n_choices, **config_controller)

        learn(sess, controller, X, y, **config_training)


if __name__ == "__main__":
    main()
