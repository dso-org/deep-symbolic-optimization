"""Defines main training loop for deep symbolic regression."""

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
from dsr.utils import MaxUniquePriorityQueue
from dsr.language_model import LanguageModelPrior
from dsr.task import make_task

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.random.set_random_seed(0)

# Work for multiprocessing pool
def work(p):
    return p.optimize()


def learn(sess, controller, logdir="./log", n_epochs=None, n_samples=1e6,
          batch_size=1000,
          complexity="length", complexity_weight=0.001,
          const_optimizer="minimize", const_params=None,
          alpha=0.1, epsilon=0.01, num_cores=1,
          verbose=True, summary=True, output_file=None, save_all_r=False,
          baseline="ewma_R", b_jumpstart=True, early_stopping=False,
          threshold=1e-12, debug=0, env_params=None):

    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.
    
    controller : dsr.controller.Controller
        Controller object used to generate Programs.

    logdir : str, optional
        Name of log directory.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

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

    save_all_r : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the sample (1-epsilon)-quantile of the batch.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if a threshold is reached.

    threshold : float, optional
        NMSE threshold to stop early if a threshold is reached.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by base_r).
    """

    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    # Create the summary writer
    if summary:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        summary_dir = os.path.join("summary", timestamp)
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Create log file
    if output_file is not None:
        os.makedirs(logdir, exist_ok=True)
        output_file = os.path.join(logdir, output_file)
        prefix, _ = os.path.splitext(output_file)
        all_r_output_file = "{}_all_r.npy".format(prefix)
        with open(output_file, 'w') as f:
            # r_best : Maximum across all iterations so far
            # r_max : Maximum across this iteration's batch
            # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
            # r_avg_sub : Average across this iteration's epsilon-subset batch
            f.write("nmse_best,nmse_min,nmse_avg_full,nmse_avg_sub,base_r_best,base_r_max,base_r_avg_full,base_r_avg_sub,r_best,r_max,r_avg_full,r_avg_sub,l_avg_full,l_avg_sub,ewma\n")

    # TBD: REFACTOR
    # Set the complexity functions
    Program.set_complexity_penalty(complexity, complexity_weight)

    # TBD: REFACTOR
    # Set the constant optimizer
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    # Initialize compute graph
    sess.run(tf.global_variables_initializer())

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
    if controller.pqt and k is not None and k > 0:
        from collections import deque
        priority_queue = MaxUniquePriorityQueue(capacity=k)
    else:
        priority_queue = None

    if debug >= 1:
        print("\nInitial parameter means:")
        print_var_means()

    # Main training loop
    nmse_best = np.inf
    base_r_best = -np.inf
    r_best = -np.inf
    prev_r_best = None
    prev_base_r_best = None
    ewma = None if b_jumpstart else 0.0 # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    all_r = np.zeros(shape=(n_epochs, batch_size), dtype=np.float32)
    #Trun on or off dsp option
    if Program.set_dsp:
        dsp = True
    else:
        dsp = False

    for step in range(n_epochs):

        # Sample batch of expressions from controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors = controller.sample(batch_size)

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

        # Retrieve metrics
        nmse = np.array([p.nmse for p in programs]) # NOTE: This adds execute() computation that might not be needed
        base_r = np.array([p.base_r for p in programs])
        r = np.array([p.r for p in programs])
        l = np.array([len(p.traversal) for p in programs])
        all_r[step] = base_r

        # Collect full-batch statistics
        if not dsp:
            nmse_min = np.min(nmse)
            nmse_best = min(nmse_min, nmse_best)
            nmse_avg_full = np.mean(nmse)
        base_r_max = np.max(base_r)
        base_r_best = max(base_r_max, base_r_best)
        base_r_avg_full = np.mean(base_r)
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_avg_full = np.mean(r)
        l_avg_full = np.mean(l)

        # Risk-seeking policy gradient: only train on top epsilon fraction of sampled expressions
        if epsilon is not None and epsilon < 1.0:
            n_keep = int(epsilon * batch_size) # Number of top indices to keep
            keep = np.zeros(shape=(batch_size,), dtype=bool)
            keep[np.argsort(r)[-n_keep:]] = True
            actions = actions[keep, :]
            obs = [o[keep, :] for o in obs]
            priors = priors[keep, :, :]
            programs = list(compress(programs, keep))
            if not dsp:
                nmse = nmse[keep]
            base_r = base_r[keep]
            r = r[keep]
            l = l[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        if baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else alpha*np.mean(r) + (1 - alpha)*ewma
            b = ewma
        elif baseline == "R_e":
            ewma = -1
            b = np.min(r)
        elif baseline == "ewma_R_e":
            ewma = np.min(r) if ewma is None else alpha*np.min(r) + (1 - alpha)*ewma
            b = ewma
        elif baseline == "combined":
            ewma = np.mean(r) - np.min(r) if ewma is None else alpha*(np.mean(r) - np.min(r)) + (1 - alpha)*ewma
            b = np.min(r) + ewma

        # Collect sub-batch statistics and write output
        if output_file is not None:
            if not dsp:
                nmse_avg_sub = np.mean(nmse)
            base_r_avg_sub = np.mean(base_r)
            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            if not dsp:
                stats = np.array([[
                             nmse_best,
                             nmse_min,
                             nmse_avg_full,
                             nmse_avg_sub,
                             base_r_best,
                             base_r_max,
                             base_r_avg_full,
                             base_r_avg_sub,
                             r_best,
                             r_max,
                             r_avg_full,
                             r_avg_sub,
                             l_avg_full,
                             l_avg_sub,
                             ewma
                             ]], dtype=np.float32)
            else:
                stats = np.array([[
                             base_r_best,
                             base_r_max,
                             base_r_avg_full,
                             base_r_avg_sub,
                             r_best,
                             r_max,
                             r_avg_full,
                             r_avg_sub,
                             l_avg_full,
                             l_avg_sub,
                             ewma
                             ]], dtype=np.float32)
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
                "obs" : [o[i] for o in obs],
                "priors" : priors[i],
                "masks" : mask[i]
            }
            # Always push unique item if the queue isn't full
            priority_queue.push(score, item, extra_data)

        # Train the controller
        summaries = controller.train_step(r, b, actions, obs, priors, mask, priority_queue)
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
                    if dsp :
                        p_r_best.dsp_evaluation(step)
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


        # Early stopping only in dsr
        if not dsp:
            if early_stopping and p_base_r_best.nmse < threshold:
                all_r = all_r[:(step + 1)]
                print("Fitness exceeded threshold; breaking early.")
                break

        # print("Step: {}, Loss: {:.6f}, baseline: {:.6f}, r: {:.6f}".format(step, loss, b, np.mean(r)))
        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))
            # print("Neglogp of ground truth action:", controller.neglogp(ground_truth_actions, ground_truth_mask)[0])

        if debug >= 2:
            print("\nParameter means after step {} of {}:".format(step+1, n_epochs))
            print_var_means()

    if save_all_r:
        with open(all_r_output_file, 'ab') as f:
            np.save(f, all_r)

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


# TBD: Should add a test instead of a main function
def main():
    """
    Loads the config file, creates the task and controller, and starts the
    training loop.
    """

    # Load the config file
    config_filename = 'config.json'
    with open(config_filename, encoding='utf-8') as f:
        config = json.load(f)

    config_task = config["task"]                # Task specification hyperparameters
    config_training = config["training"]        # Training hyperparameters
    config_controller = config["controller"]    # Controller hyperparameters
    config_language_model_prior = config["language_model"]            # Language model hyperparameters

    # Define the task
    reward_function, function_set, n_input_var = make_task(**config_task)
    Program.set_reward_function(reward_function)
    Program.set_library(function_set, n_input_var)
    Program.set_execute()
    # print("Ground truth expression:\n{}".format(indent(task.dataset.pretty(), '\t')))

    with tf.Session() as sess:
        # Instantiate the controller
        language_model_prior = LanguageModelPrior(dataset.function_set, dataset.n_input_var, **config_language_model_prior)
        controller = Controller(sess, debug=config_training["debug"], summary=config_training["summary"], language_model_prior=language_model_prior, **config_controller)
        learn(sess, controller, **config_training)


if __name__ == "__main__":

    if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
        import cProfile
        cProfile.run('main()', sort='cumtime')
    else:
        main()
