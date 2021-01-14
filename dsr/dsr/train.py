"""Defines main training loop for deep symbolic regression."""

import os
import multiprocessing
from itertools import compress
from datetime import datetime
from collections import defaultdict

import tensorflow as tf
import pandas as pd
import numpy as np

from dsr.program import Program, from_tokens
from dsr.utils import empirical_entropy, is_pareto_efficient, setup_output_files
from dsr.memory import Batch, make_queue

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.random.set_random_seed(0)


# Work for multiprocessing pool: optimize constants and compute reward
def work(p):
    optimized_constants = p.optimize()
    return optimized_constants, p.base_r


def hof_work(p):
    return [p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]


def pf_work(p):
    return [p.complexity_eureqa, p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]


def learn(sess, controller, pool,
          logdir="./log", n_epochs=None, n_samples=1e6,
          batch_size=1000, complexity="length", complexity_weight=0.001,
          const_optimizer="minimize", const_params=None, alpha=0.1,
          epsilon=0.01, n_cores_batch=1, verbose=True, summary=True,
          output_file=None, save_all_r=False, baseline="ewma_R",
          b_jumpstart=True, early_stopping=False, hof=10, eval_all=False,
          pareto_front=False, debug=0):
    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.

    controller : dsr.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

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

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

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
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by base_r).
    """

    # Config assertions and warnings
    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    # Create the summary writer
    if summary:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        summary_dir = os.path.join("summary", timestamp)
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Create log file
    if output_file is not None:
        all_r_output_file, hof_output_file, pf_output_file = setup_output_files(logdir, output_file)
    else:
        all_r_output_file = hof_output_file = pf_output_file = None

    # Set the complexity functions
    Program.set_complexity_penalty(complexity, complexity_weight)

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
                print(var.name, "mean:", val.mean(),"var:", val.var())

    # Create the pool of workers, if pool is not already given
    if pool is None:
        if n_cores_batch == -1:
            n_cores_batch = multiprocessing.cpu_count()
        if n_cores_batch > 1:
            pool = multiprocessing.Pool(n_cores_batch)

    # Create the priority queue
    k = controller.pqt_k
    if controller.pqt and k is not None and k > 0:
        priority_queue = make_queue(priority=True, capacity=k)
    else:
        priority_queue = None

    if debug >= 1:
        print("\nInitial parameter means:")
        print_var_means()

    base_r_history = None

    # Main training loop
    p_final = None
    base_r_best = -np.inf
    r_best = -np.inf
    prev_r_best = None
    prev_base_r_best = None
    ewma = None if b_jumpstart else 0.0 # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    all_r = np.zeros(shape=(n_epochs, batch_size), dtype=np.float32)

    for step in range(n_epochs):

        # Set of str representations for all Programs ever seen
        s_history = set(Program.cache.keys())

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
            # first generated serially. Programs that need optimizing are
            # optimized optimized in parallel. Since multiprocessing operates on
            # copies of programs, we manually set the optimized constants and
            # base reward after the pool joins.
            programs = [from_tokens(a, optimize=False) for a in actions]

            # Filter programs that have not yet computed base_r
            programs_to_optimize = list(set([p for p in programs if "base_r" not in p.__dict__]))

            # Optimize and compute base_r
            results = pool.map(work, programs_to_optimize)
            for (optimized_constants, base_r), p in zip(results, programs_to_optimize):
                p.set_constants(optimized_constants)
                p.base_r = base_r

        # Retrieve metrics
        base_r = np.array([p.base_r for p in programs])
        r = np.array([p.r for p in programs])
        l = np.array([len(p.traversal) for p in programs])
        s = [p.str for p in programs] # Str representations of Programs
        invalid = np.array([p.invalid for p in programs], dtype=bool)
        all_r[step] = base_r

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]
            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]

        # Update reward history
        if base_r_history is not None:
            for p in programs:
                key = p.str
                if key in base_r_history:
                    base_r_history[key].append(p.base_r)
                else:
                    base_r_history[key] = [p.base_r]

        # Collect full-batch statistics
        base_r_max = np.max(base_r)
        base_r_best = max(base_r_max, base_r_best)
        base_r_avg_full = np.mean(base_r)
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_avg_full = np.mean(r)
        l_avg_full = np.mean(l)
        a_ent_full = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
        n_unique_full = len(set(s))
        n_novel_full = len(set(s).difference(s_history))
        invalid_avg_full = np.mean(invalid)

        # Risk-seeking policy gradient: train on top epsilon fraction of samples
        if epsilon is not None and epsilon < 1.0:
            quantile = np.quantile(r, 1 - epsilon, interpolation="higher")
            keep = base_r >= quantile
            base_r = base_r[keep]
            r_train = r = r[keep]
            programs = list(compress(programs, keep))
            l = l[keep]
            s = list(compress(s, keep))
            invalid = invalid[keep]
            actions = actions[keep, :]
            obs = [o[keep, :] for o in obs]
            priors = priors[keep, :, :]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        if baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else alpha*np.mean(r) + (1 - alpha)*ewma
            b_train = ewma
        elif baseline == "R_e": # Default
            ewma = -1
            b_train = quantile

        # Collect sub-batch statistics and write output
        if output_file is not None:
            base_r_avg_sub = np.mean(base_r)
            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            a_ent_sub = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
            n_unique_sub = len(set(s))
            n_novel_sub = len(set(s).difference(s_history))
            invalid_avg_sub = np.mean(invalid)
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
                         ewma,
                         n_unique_full,
                         n_unique_sub,
                         n_novel_full,
                         n_novel_sub,
                         a_ent_full,
                         a_ent_sub,
                         invalid_avg_full,
                         invalid_avg_sub
                         ]], dtype=np.float32)
            with open(os.path.join(logdir, output_file), 'ab') as f:
                np.savetxt(f, stats, delimiter=',')

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), controller.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r)

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, programs)
            pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
        else:
            pqt_batch = None

        # Train the controller
        summaries = controller.train_step(b_train, sampled_batch, pqt_batch)
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

        # Stop if early stopping criteria is met
        if eval_all and any(success):
            all_r = all_r[:(step + 1)]
            print("Early stopping criteria met; breaking early.")
            break
        if early_stopping and p_base_r_best.evaluate.get("success"):
            all_r = all_r[:(step + 1)]
            print("Early stopping criteria met; breaking early.")
            break

        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))

        if debug >= 2:
            print("\nParameter means after step {} of {}:".format(step+1, n_epochs))
            print_var_means()

    if save_all_r:
        with open(all_r_output_file, 'ab') as f:
            np.save(f, all_r)

    # Save the hall of fame
    if hof is not None and hof > 0:
        programs = list(Program.cache.values()) # All unique Programs found during training

        base_r = [p.base_r for p in programs]
        i_hof = np.argsort(base_r)[-hof:][::-1] # Indices of top hof Programs
        hof = [programs[i] for i in i_hof]

        if verbose:
            print("Evaluating the hall of fame...")
        if pool is not None:
            results = pool.map(hof_work, hof)
        else:
            results = list(map(hof_work, hof))

        eval_keys = list(results[0][-1].keys())
        columns = ["r", "base_r", "count", "expression", "traversal"] + eval_keys
        hof_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(hof_results, columns=columns)
        if hof_output_file is not None:
            print("Saving Hall of Fame to {}".format(hof_output_file))
            df.to_csv(hof_output_file, header=True, index=False)
        
    # Print error statistics of the cache
    n_invalid = 0
    error_types = defaultdict(lambda : 0)
    error_nodes = defaultdict(lambda : 0)
    for p in Program.cache.values():
        if p.invalid:
            n_invalid += p.count
            error_types[p.error_type] += p.count
            error_nodes[p.error_node] += p.count
    if n_invalid > 0:
        total_samples = (step + 1)*batch_size # May be less than n_samples if breaking early
        print("Invalid expressions: {} of {} ({:.1%}).".format(n_invalid, total_samples, n_invalid/total_samples))
        print("Error type counts:")
        for error_type, count in error_types.items():
            print("  {}: {} ({:.1%})".format(error_type, count, count/n_invalid))
        print("Error node counts:")
        for error_node, count in error_nodes.items():
            print("  {}: {} ({:.1%})".format(error_node, count, count/n_invalid))

    # Print the priority queue at the end of training
    if verbose and priority_queue is not None:
        for i, item in enumerate(priority_queue.iter_in_order()):
            print("\nPriority queue entry {}:".format(i))
            p = Program.cache[item[0]]
            p.print_stats()

    # Compute the pareto front
    if pareto_front:
        if verbose:
            print("Evaluating the pareto front...")
        all_programs = list(Program.cache.values())
        costs = np.array([(p.complexity_eureqa, -p.r) for p in all_programs])
        pareto_efficient_mask = is_pareto_efficient(costs) # List of bool
        pf = list(compress(all_programs, pareto_efficient_mask))
        pf.sort(key=lambda p : p.complexity_eureqa) # Sort by complexity

        if pool is not None:
            results = pool.map(pf_work, pf)
        else:
            results = list(map(pf_work, pf))

        eval_keys = list(results[0][-1].keys())
        columns = ["complexity", "r", "base_r", "count", "expression", "traversal"] + eval_keys
        pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(pf_results, columns=columns)
        if pf_output_file is not None:
            print("Saving Pareto Front to {}".format(pf_output_file))
            df.to_csv(pf_output_file, header=True, index=False)

        # Look for a success=True case within the Pareto front
        for p in pf:
            if p.evaluate.get("success"):
                p_final = p
                break

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_base_r_best
    result = {
        "r" : p.r,
        "base_r" : p.base_r,
    }
    result.update(p.evaluate)
    result.update({
        "expression" : repr(p.sympy_expr),
        "traversal" : repr(p),
        "program" : p
    })

    return result
