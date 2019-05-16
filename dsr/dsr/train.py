import os
import json
from datetime import datetime
import itertools

import tensorflow as tf
import numpy as np

from dsr.controller import Controller
from dsr.program import Program
from dsr.dataset import Dataset


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
            programs = list(itertools.compress(programs, cutoff))
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
            print("New best expression: {} (reward = {})".format(programs[index], best))


def main():

    # Load the config file
    config_filename = 'config.json'
    with open(config_filename, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]          # Problem specification parameters
    config_training = config["training"]        # Training hyperparameters
    config_controller = config["controller"]    # Controller hyperparameters

    # HACK: Overriding values in config_dataset until reading in benchmarks is complete
    config_dataset.update({
        "traversal" : ["Add","Add","Add","Add","x1","x2","x3","x4","x5"],
        # "traversal" : ["Mul","sin","x1","cos","Mul","x1","x2"],
        "n_input_var" : 10,
        "train_spec" : {
            "x1" : {"U" : [-5, 5, 600]},
            "x2" : {"U" : [-5, 5, 600]},
            "x3" : {"U" : [-5, 5, 600]},
            "x4" : {"U" : [-5, 5, 600]},
            "x5" : {"U" : [-5, 5, 600]},
            "x6" : {"U" : [-5, 5, 600]},
            "x7" : {"U" : [-5, 5, 600]},
            "x8" : {"U" : [-5, 5, 600]},
            "x9" : {"U" : [-5, 5, 600]},
            "x10" : {"U" : [-5, 5, 600]},
        },
        "test_spec" : None
    })

    # Define the library
    Program.set_library(config_dataset["operators"], config_dataset["n_input_var"])
    
    # Create the dataset
    dataset = Dataset(**config_dataset)
    X, y = dataset.X_train, dataset.y_train
    n_choices = len(Program.library)

    with tf.Session() as sess:

        # Instantiate the controller
        controller = Controller(sess, n_choices=n_choices, **config_controller)

        learn(sess, controller, X, y, **config_training)


if __name__ == "__main__":
    main()