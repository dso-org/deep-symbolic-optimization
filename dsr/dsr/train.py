"""Defines main training loop for deep symbolic regression."""

import os
import sys
import json
import multiprocessing
from itertools import compress
from datetime import datetime
from textwrap import indent
from collections import defaultdict

import tensorflow as tf
import pandas as pd
import numpy as np

from dsr.controller import Controller
from dsr.program import Program, from_tokens
from dsr.utils import MaxUniquePriorityQueue, empirical_entropy, is_pareto_efficient
from dsr.language_model import LanguageModelPrior

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


def learn(sess, controller, pool, logdir="./log", n_epochs=None, n_samples=1e6,
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
        (4) "combined" = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the sample (1-epsilon)-quantile of the batch.

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
    if epsilon is not None and batch_size * epsilon < 1:
        print("WARNING: batch_size * epsilon < 1. Risk-seeking will not be used.")

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
        hof_output_file = "{}_hof.csv".format(prefix)
        pf_output_file = "{}_pf.csv".format(prefix)
        with open(output_file, 'w') as f:
            # r_best : Maximum across all iterations so far
            # r_max : Maximum across this iteration's batch
            # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
            # r_avg_sub : Average across this iteration's epsilon-subset batch
            # n_unique_* : Number of unique Programs in batch
            # n_novel_* : Number of never-before-seen Programs per batch
            # a_ent_* : Empirical positional entropy across sequences averaged over positions 
            # invalid_avg_* : Fraction of invalid Programs per batch
            headers = ["base_r_best",
                       "base_r_max",
                       "base_r_avg_full",
                       "base_r_avg_sub",
                       "r_best",
                       "r_max",
                       "r_avg_full",
                       "r_avg_sub",
                       "l_avg_full",
                       "l_avg_sub",
                       "ewma",
                       "n_unique_full",
                       "n_unique_sub",
                       "n_novel_full",
                       "n_novel_sub",
                       "a_ent_full",
                       "a_ent_sub",
                       "invalid_avg_full",
                       "invalid_avg_sub"]
            f.write("{}\n".format(",".join(headers)))

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

    # Create the pool of workers, if pool is not already given
    if pool is None:
        if n_cores_batch == -1:
            n_cores_batch = multiprocessing.cpu_count()
        if n_cores_batch > 1:
            pool = multiprocessing.Pool(n_cores_batch)            

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

    # For stochastic Programs, store each base_r computation for each unique traversal
    if Program.stochastic:
        base_r_history = {} # Dict from Program str to list of base_r values
        # It's not really clear stochastic Programs with const should enter the hof
        assert "const" not in Program.library, "Constant tokens not yet supported with stochastic Programs"
    else:
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
        s_history = set(base_r_history.keys() if Program.stochastic else Program.cache.keys())

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
            # TBD: Refactor with needs_optimizing flag or similar?
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
            if any(success):
                all_r = all_r[:(step + 1)]
                p_final = programs[success.index(True)]
                print("Early stopping criteria met; breaking early.")
                break

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

        # Risk-seeking policy gradient: only train on top epsilon fraction of sampled expressions
        if epsilon is not None and epsilon < 1.0:
            n_keep = int(epsilon * batch_size) # Number of top indices to keep
            keep = np.zeros(shape=(batch_size,), dtype=bool)
            keep[np.argsort(r)[-n_keep:]] = True
            actions = actions[keep, :]
            obs = [o[keep, :] for o in obs]
            priors = priors[keep, :, :]
            programs = list(compress(programs, keep))
            base_r = base_r[keep]
            r = r[keep]
            l = l[keep]
            s = list(compress(s, keep))
            invalid = invalid[keep]

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
                "masks" : mask[i],
                "program" : p
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

        # For stochastic Programs, average each unique Program's base_r_history,
        if Program.stochastic:

            # Define a helper function to generate a Program from its tostring() value
            def from_token_string(str_tokens, optimize):
                tokens = np.fromstring(str_tokens, dtype=np.int32)
                return from_tokens(tokens, optimize=optimize)

            # Generate each unique Program and manually set its base_r to the average of its base_r_history
            keys = base_r_history.keys() # str_tokens for each unique Program
            vals = base_r_history.values() # base_r histories for each unique Program
            programs = [from_token_string(str_tokens, optimize=False) for str_tokens in keys]
            for p, base_r in zip(programs, vals):
                p.base_r = np.mean(base_r)
                p.count = len(base_r) # HACK
                _ = p.r # HACK: Need to cache reward here (serially) because pool doesn't know the complexity_function

        # For deterministic Programs, just use the cache
        else:
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
        df.to_csv(hof_output_file, header=True, index=False)

    if pool is not None:
        pool.close()

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
            item[1]["program"].print_stats()

    # Compute the pareto front
    if pareto_front:
        if verbose:
            print("Evaluating the pareto front...")
        all_programs = list(Program.cache.values())
        costs = np.array([(p.complexity_eureqa, -p.r) for p in all_programs])
        pareto_efficient_mask = is_pareto_efficient(costs) # List of bool
        pf = list(compress(all_programs, pareto_efficient_mask))

        if pool is not None:
            results = pool.map(pf_work, pf)
        else:
            results = list(map(pf_work, pf))

        eval_keys = list(results[0][-1].keys())
        columns = ["complexity", "r", "base_r", "count", "expression", "traversal"] + eval_keys
        pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(pf_results, columns=columns)
        df.to_csv(pf_output_file, header=True, index=False)

    # Return statistics of best Program
    p = p_final if p_final is not None else p_base_r_best
    result = {
        "r" : p.r,
        "base_r" : p.base_r,
    }
    result.update(p.evaluate)
    result.update({
        "expression" : repr(p.sympy_expr),
        "traversal" : repr(p)
        })
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
    config_language_model_prior = config["language_model_prior"]            # Language model hyperparameters

    # Define the task
    from dsr.task import set_task
    reward_function, eval_function, function_set, n_input_var = make_task(**config_task)
    Program.set_reward_function(reward_function)
    Program.set_eval_function(eval_function)
    Program.set_library(function_set, n_input_var)
    Program.set_execute()

    with tf.Session() as sess:
        # Instantiate the controller
        language_model_prior = LanguageModelPrior(function_set, n_input_var, **config_language_model_prior)
        controller = Controller(sess, debug=config_training["debug"], summary=config_training["summary"], language_model_prior=language_model_prior, **config_controller)
        learn(sess, controller, **config_training)


if __name__ == "__main__":

    if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
        import cProfile
        cProfile.run('main()', sort='cumtime')
    else:
        main()
