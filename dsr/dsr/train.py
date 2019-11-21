import os
import sys
import json
import multiprocessing
from itertools import compress
from datetime import datetime
from textwrap import indent

import tensorflow as tf
import pandas as pd
import numpy as np

from dsr.controller import Controller
from dsr.program import Program, from_tokens
from dsr.dataset import Dataset
from dsr.utils import MaxUniquePriorityQueue


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
          verbose=True, summary=True, output_file=None, b_jumpstart=True,
          early_stopping=False, threshold=1e-12):
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

    output_file : str, optional
        Filename to write results for each iteration.

    b_jumpstart : bool, optional
        Whether baseline start at average reward for the first iteration. If
        False, it starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if a threshold is reached.

    threshold : float, optional
        NMSE threshold to stop early if a threshold is reached.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by base_r).
    """

    # Create the summary writer
    if summary:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        summary_dir = os.path.join("summary", timestamp)
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Create log file
    if output_file is not None:
        logdir = os.path.join("log", logdir)
        os.makedirs(logdir, exist_ok=True)
        output_file = os.path.join(logdir, output_file)
        with open(output_file, 'w') as f:
            # r_best : Maximum across all iterations so far
            # r_max : Maximum across this iteration's batch
            # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
            # r_avg_sub : Average across this iteration's epsilon-subset batch
            f.write("base_r_best,base_r_max,base_r_avg_full,base_r_avg_sub,r_best,r_max,r_avg_full,r_avg_sub,l_avg_full,l_avg_sub,baseline,nmse_best,nmse_avg_full\n")

    # Set the reward and complexity functions
    reward_params = reward_params if reward_params is not None else []
    Program.set_reward_function(reward, *reward_params)
    Program.set_complexity_penalty(complexity, complexity_weight)

    # Set the constant optimizer
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    # Initialize compute graph
    sess.run(tf.global_variables_initializer())

    debug = False
    if debug:
        tvars = tf.trainable_variables()
        def print_var_means():
            tvars_vals = sess.run(tvars)
            for var, val in zip(tvars, tvars_vals):
                print(var.name, val.mean())        

    # Create the pool of workers
    pool = None
    if "const" in Program.library:
        if num_cores == -1:
            num_cores = multiprocessing.cpu_count()
        if num_cores > 1:
            pool = multiprocessing.Pool(num_cores)

    # Create the priority queue
    k = controller.pqt_k
    if k is not None and k > 0:
        from collections import deque
        priority_queue = MaxUniquePriorityQueue(capacity=k)
    else:
        priority_queue = None

    # Only needed for batch statistics log
    if output_file is not None:
        nmse_best = np.inf

    # Main training loop
    # max_count = 1    
    r_best = -np.inf
    base_r_best = -np.inf
    prev_r_best = None
    prev_base_r_best = None
    b = None if b_jumpstart else 0.0 # Baseline used for control variates
    for step in range(n_epochs):

        if debug:
            print_var_means()

        # Sample batch of expressions from controller
        actions, inputs, priors = controller.sample(batch_size) # Shape: (batch_size, max_length), (batch_size, max_length, n_inputs)

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
            for optimized_constants, p in zip(results, programs_to_optimize):
                p.set_constants(optimized_constants)

        # Retrieve the rewards
        r = np.array([p.r for p in programs])
        base_r = np.array([p.base_r for p in programs])
        l = np.array([len(p.traversal) for p in programs])

        # Collect full-batch statistics
        base_r_max = np.max(base_r)
        base_r_best = max(base_r_max, base_r_best)
        base_r_avg_full = np.mean(base_r)
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_avg_full = np.mean(r)
        l_avg_full = np.mean(l)

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
        cutoff = None
        if epsilon is not None and epsilon < 1.0:
            cutoff = r >= np.percentile(r, 100 - int(100*epsilon))
            actions = actions[cutoff, :]
            inputs = inputs[cutoff, :, :]
            priors = priors[cutoff, :, :]
            programs = list(compress(programs, cutoff))
            r = r[cutoff]
            l = l[cutoff]
            base_r = base_r[cutoff]

        # Clip lower bound of rewards to prevent NaNs in gradient descent
        if reward in ["neg_mse", "neg_nmse", "neg_nrmse"]:
            r = np.clip(r, -1e6, np.inf)

        # Compute baseline (EWMA of average reward)
        b = np.mean(r) if b is None else alpha*np.mean(r) + (1 - alpha)*b

        # Collect sub-batch statistics and write output
        if output_file is not None:
            base_r_avg_sub = np.mean(base_r)
            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            nmse = np.array([p.nmse for p in programs])
            nmse_min = np.min(nmse)
            nmse_best = min(nmse_min, nmse_best)
            nmse_avg_full = np.mean(nmse)
            stats = np.array([[base_r_best,
                             base_r_max,
                             base_r_avg_full,
                             base_r_avg_sub,
                             r_best,
                             r_max,
                             r_avg_full,
                             r_avg_sub,
                             l_avg_full,
                             l_avg_sub,
                             b,
                             nmse_best,
                             nmse_avg_full]], dtype=np.float32)
            with open(output_file, 'ab') as f:
                np.savetxt(f, stats, delimiter=',')

        # Compute actions mask
        mask = np.zeros_like(actions, dtype=np.float32) # Shape: (batch_size, max_length)
        for i,p in enumerate(programs):
            length = min(len(p.traversal), controller.max_length)
            mask[i, :length] = 1.0

        # Update the priority queue
        # NOTE: Updates with at most one expression per batch
        if priority_queue is not None:
            i = np.argmax(r)
            p = programs[i]
            score = p.r
            item = p.tokens.tostring()
            extra_data = {
                "actions" : actions[i],
                "inputs" : inputs[i],
                "priors" : priors[i],
                "masks" : mask[i]
            }
            # Always push unique item if the queue isn't full
            priority_queue.push(score, item, extra_data)

        # Train the controller
        summaries = controller.train_step(r, b, actions, inputs, priors, mask, priority_queue)
        if summary:
            writer.add_summary(summaries, step)
            writer.flush()

        # Update new best expression
        new_r_best = False
        new_base_r_best = False
        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = programs[np.argmax(r)]
        if prev_base_r_best is None or base_r_max > prev_base_r_best:
            new_base_r_best = True
            p_base_r_best = programs[np.argmax(base_r)]
        prev_r_best = r_best
        prev_base_r_best = base_r_best

        # Print new best expression
        if verbose:
            if new_r_best and new_base_r_best:
                if p_r_best == p_base_r_best:
                    print("\nNew best overall")
                    p_r_best.print_stats()
                else:
                    print("\nNew best reward")
                    p_r_best.print_stats()
                    print("...and new best base reward")
                    p_base_r_best.print_stats()
            elif new_r_best:
                print("\nNew best reward")
                p_r_best.print_stats()
            elif new_base_r_best:
                print("\nNew best base reward")
                p_base_r_best.print_stats()

        # Early stopping
        if early_stopping and p_base_r_best.nmse < threshold:
            print("Fitness exceeded threshold; breaking early.")
            break

        # print("Step: {}, Loss: {:.6f}, baseline: {:.6f}, r: {:.6f}".format(step, loss, b, np.mean(r)))
        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))
            # print("Neglogp of ground truth action:", controller.neglogp(ground_truth_actions, ground_truth_mask)[0])

    if pool is not None:
        pool.close()

    p = p_base_r_best
    result = {
            # "p_r_best.r" : p_r_best.r,
            # "p_r_best.base_r" : p_r_best.base_r
            # "p_r_best_expression" : repr(p_r_best.sympy_expr),
            # "p_r_best_traversal" : repr(p_r_best),
            # "p_base_r_best.r" : p_base_r_best.r,
            # "p_base_r_best.base_r" : p_base_r_best.base_r
            # "p_base_r_best_expression" : repr(p_base_r_best.sympy_expr),
            # "p_base_r_best_traversal" : repr(p_base_r_best),
            "nmse" : p.nmse, # Final performance metric
            "r" : p.r,
            "base_r" : p.base_r,
            "r_test" : p.r_test,
            "base_r_test" : p.base_r_test,
            "r_noiseless" : p.r_noiseless,
            "base_r_noiseless" : p.base_r_noiseless,
            "r_test_noiseless" : p.r_test_noiseless,
            "base_r_test_noiseless" : p.base_r_test_noiseless,
            "expression" : repr(p.sympy_expr),
            "traversal" : repr(p)
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

    # Define the dataset and library
    dataset = Dataset(**config_dataset)
    Program.set_training_data(dataset)
    Program.set_library(dataset.function_set, dataset.n_input_var)
    print("Ground truth expression:\n{}".format(indent(dataset.pretty(), '\t')))

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
