import os
import sys
import json
import multiprocessing
from itertools import compress
from datetime import datetime
from textwrap import indent

import tensorflow as tf
import numpy as np

from dsr.controller import Controller
from dsr.program import Program, from_tokens
from dsr.dataset import Dataset


# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Set TensorFlow seed
tf.random.set_random_seed(0)

# Work for multiprocessing pool
def work(p):
    return p.optimize()


def learn(sess, controller, logdir=".", n_epochs=1000, batch_size=1000,
          reward="neg_mse", reward_params=None, complexity="length",
          complexity_weight=0.001, const_optimizer="minimize",
          const_params=None, alpha=0.1, epsilon=0.01, num_cores=1,
          verbose=True, summary=True):
    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.
    
    controller : Controller
        Controller object.
    
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

    num_cores : int, optional
        Number of cores to use for optimizing programs. If -1, uses
        multiprocessing.cpu_count().
    
    verbose : bool, optional
        Whether to print progress.

    summary : bool, optional
        Whether to write TensorFlow summaries.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression: 'r' is the reward,
        'traversal' is the serialized Program, and 'expresion' is the pretty-
        printed sympy-simplified expression
    """

    # Create the summary writer
    if summary:
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

    # Create the pool of workers
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()
    if num_cores > 1:
        pool = multiprocessing.Pool(num_cores)
    else:
        pool = None

    # Main training loop
    max_count = 1
    max_r = -np.inf # Best reward
    best = None # Best program
    b = None # Baseline used for control variates
    for step in range(n_epochs):

        # Sample batch of expressions from controller
        actions = controller.sample(batch_size) # Shape: (batch_size, max_length)

        # Instantiate, optimize, and evaluate expressions
        if pool is None:
            programs = [from_tokens(a, optimize=True) for a in actions]
        else:
            # To prevent interfering with the cache, un-optimized programs are
            # first generated serially. The resulting set is optimized in
            # parallel. Since multiprocessing operates on copies of programs,
            # we manually set the optimized constants and base reward after the
            # pool joins.
            programs = [from_tokens(a, optimize=False) for a in actions]
            programs_to_optimize = list(set([p for p in programs if p.base_r is None]))
            results = pool.map(work, programs_to_optimize)
            for pair, p in zip(results, programs_to_optimize):
                optimized_constants, base_r = pair
                p.set_constants(optimized_constants)
                p.base_r = base_r
        
        # Retrieve the rewards
        r = np.array([p.r for p in programs])

        # # Show new commonest expression
        # # Note: This should go before epsilon heuristic
        # counts = np.array([p.count for p in programs])
        # if max(counts) > max_count:
        #     max_count = max(counts)
        #     commonest = programs[np.argmax(counts)]            
        #     if verbose:                
        #         print("\nNew commonest expression")
        #         commonest.print_stats()

        # Heuristic: Only train on top epsilon fraction of sampled expressions
        if epsilon is not None and epsilon < 1.0:
            cutoff = r >= np.percentile(r, 100 - int(100*epsilon))
            actions = actions[cutoff, :]
            programs = list(compress(programs, cutoff))
            r = r[cutoff]

        # Compute baseline (EWMA of average reward)
        b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b

        # Compute actions mask
        actions_mask = np.zeros_like(actions.T, dtype=np.float32) # Shape: (max_length, batch_size)
        for i,p in enumerate(programs):
            length = min(len(p.traversal), controller.max_length)
            actions_mask[:length, i] = 1.0

        # Train the controller
        loss, summaries = controller.train_step(r, b, actions, actions_mask)
        if summary:
            writer.add_summary(summaries, step)
            writer.flush()

        updated=0
        # Show new best expression
        if max(r) > max_r:
            updated=1
            max_r = max(r)
            best = programs[np.argmax(r)]
            if verbose:
                print("\nNew best expression")
                best.print_stats()

#--- as an argument, it is needed to pass aditional fields that one wishes to print that are not fields of the controller class
        controller.savestepinfo({"step":step,"traversal":programs[np.argmax(r)],"reward":r,"updated":updated})
        # print("Step: {}, Loss: {:.6f}, baseline: {:.6f}, r: {:.6f}".format(step, loss, b, np.mean(r)))
        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))
            # print("Neglogp of ground truth action:", controller.neglogp(ground_truth_actions, ground_truth_actions_mask)[0])

    if pool is not None:
        pool.close()

    result = {
            "r" : best.r,
            "expression" : repr(best.sympy_expr),
            "traversal" : repr(best)
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
    Program.set_training_data(X, y)
    print("Ground truth expression:\n{}".format(indent(dataset.pretty(), '\t')))

    # Define the library
    Program.set_library(config_dataset["operators"], X.shape[1])
    
    with tf.Session() as sess:
        # Instantiate the controller
        controller = Controller(sess, summary=config_training["summary"], **config_controller)
        learn(sess, controller, **config_training)


if __name__ == "__main__":
    
    if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
        import cProfile
        cProfile.run('main()', sort='cumtime')
    else:
        main()
