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


def learn(sess, controller, X, y, logdir=".", n_epochs=1000, batch_size=1000,
          reward="neg_mse", reward_params=None, complexity="length",
          complexity_weight=0.001, const_optimizer="minimize",
          const_params=None, alpha=0.1, epsilon=0.01, verbose=True):
    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.
    
    controller : Controller
        Controller object.
    
    X, y : np.ndarray
        Training data used for symbolic regression.
    
    logdir : str, optional
        Name of log directory.
    
    n_epochs : int, optional
        Number of epochs to train.
    
    batch_size : int, optional
        Number of sampled expressions per epoch.
    
    reward : str, optional
        Reward function name.
    
    reward_params : list of str, optional
        List of reward function parameters.
    
    complexity : str, optional
        Complexity penalty name.

    complexity_weight : float, optional
        Coefficient for complexity penalty.

    const_optimizer : str or None, optional
        Name of constant optimizer.
    
    const_params : dict, optional
        Dict of constant optimizer kwargs.
    
    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.
    
    epsilon : float, optional
        Fraction of top expressions used for training.
    
    verbose : bool, optional
        Whether to print progress.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression: 'r' is the reward,
        'traversal' is the serialized Program, and 'expresion' is the pretty-
        printed sympy-simplified expression
    """

    # Create the summary writer
    logdir = os.path.join("log", logdir)
    os.makedirs(logdir, exist_ok=True)
    logdir = "./summary/{}/".format(datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    writer = tf.summary.FileWriter(logdir, sess.graph)

    # Set the reward and complexity functions
    reward_params = reward_params if reward_params is not None else []
    Program.set_reward_function(reward, *reward_params)
    Program.set_complexity_penalty(complexity, complexity_weight)

    # Set the constant optimizer
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    # Initialize compute graph
    sess.run(tf.global_variables_initializer())        

    # Main training loop
    best_r = -np.inf # Best reward
    best_program = None # Best program
    b = None # Baseline used for control variates
    for step in range(n_epochs):

        actions = controller.sample(batch_size) # Sample batch of expressions from controller
        actions = np.squeeze(np.stack(actions, axis=-1)) # Shape (batch_size, max_length)

        # unique_actions, counts = np.unique(np.squeeze(np.stack(actions, axis=-1))[:,:5], axis=0, return_counts=True)
        # print(unique_actions.shape[0])
        
        # TBD: Parallelize
        # Instantiate, optimize, and evaluate expressions
        programs = [Program(a) for a in actions]
        programs = [p.optimize(X, y) for p in programs]
        base_r = np.array([p.base_r for p in programs])
        complexity = np.array([p.complexity for p in programs])
        r = base_r - complexity # Reward = base reward - complexity penalty

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
        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))
            # print("Neglogp of ground truth action:", controller.neglogp(ground_truth_actions, ground_truth_actions_mask)[0])

        if max(r) > best_r:
            index = np.argmax(r)
            best_r = r[index]
            best_program = programs[index]
            if verbose:
                print("\nNew best expression:")
                print("\tReward: {}".format(best_r))
                print("\tTraversal: {}".format(best_program))
                print("\tExpression:")
                print("{}\n".format(indent(best_program.pretty(), '\t  ')))

    result = {
            "r" : best_r,
            "expression" : repr(best_program.get_sympy_expr()),
            "traversal" : repr(best_program)
            }
    return result


def main():
    """
    Loads the config file, creates the library and controller, and starts the
    training loop.
    """

    # Load the config file
    config_filename = 'config.json'
    with open(config_filename, encoding='utf-8') as f:
        config = json.load(f)

    config_dataset = config["dataset"]          # Problem specification hyperparameters
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
