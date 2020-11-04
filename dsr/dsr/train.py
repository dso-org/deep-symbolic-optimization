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
from dsr.utils import MaxUniquePriorityQueue, empirical_entropy, is_pareto_efficient, Batch, setup_output_files
from dsr.language_model import LanguageModelPrior

try:
    from deap import tools
    from deap import gp
except ImportError:
    tools   = None
    gp      = None

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

# def sympy_work(p):
#     sympy_expr = p.sympy_expr
#     str_sympy_expr = repr(p.sympy_expr) if sympy_expr != "N/A" else repr(p)
#     return sympy_expr, str_sympy_expr

def learn(sess, controller, pool, gp_controller,
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
    
    all_r_size              = batch_size

    if gp_controller is not None:
        run_gp_meld             = True
        gp_verbose              = gp_controller.config_gp_meld["verbose"]
        if gp_controller.config_gp_meld["train_n"]:
            all_r_size              = batch_size+gp_controller.config_gp_meld["train_n"]
        else:
            all_r_size              = batch_size+1
    else:
        gp_controller           = None
        run_gp_meld             = False                         
        gp_verbose              = False
    
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
        all_r_output_file, hof_output_file, pf_output_file = setup_output_files(logdir, output_file)

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

    # For stochastic Tasks, store each base_r computation for each unique traversal
    if Program.task.stochastic:
        base_r_history = {} # Dict from Program str to list of base_r values
        # It's not really clear whether Programs with const should enter the hof for stochastic Tasks
        assert "const" not in Program.library, "Constant tokens not yet supported with stochastic Tasks."
        assert not pareto_front, "Pareto front not supported with stochastic Tasks."
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
    all_r = np.zeros(shape=(n_epochs, all_r_size), dtype=np.float32)
    
    nevals              = 0
    program_val_log     = []

    for step in range(n_epochs):

        if gp_verbose:
            print("************************************************************************")
            print("STEP {}".format(step))
            print("************************")

        # Set of str representations for all Programs ever seen
        s_history = set(base_r_history.keys() if Program.task.stochastic else Program.cache.keys())

        # Sample batch of expressions from controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors                = controller.sample(batch_size)
        
        nevals += batch_size

        if run_gp_meld:
            '''
                Given the set of 'actions' we have so far, we will use them as a prior seed into 
                the GP controller. It will take care of conversion to its own population data
                structures. It will return programs, observations, actions that are compat with 
                the current way we do things in train.py.
            '''            
            deap_programs, deap_obs, deap_actions, deap_priors = gp_controller(actions)
            
            nevals += gp_controller.nevals
            
            if gp_verbose:           
                print("************************")
                print("Number of Evaluations: {}".format(nevals))
                print("************************")
                print("Deap Programs:")
                deap_programs[0].print_stats()
                print("************************")
                                               
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

        # If we run GP, insert GP Program, actions, priors (blank) and obs.
        # We may option later to return these to the controller.  
        if run_gp_meld:
            programs    = programs + deap_programs
            actions     = np.append(actions, deap_actions, axis=0)
            obs         = [np.append(obs[0], deap_obs[0], axis=0),
                           np.append(obs[1], deap_obs[1], axis=0),
                           np.append(obs[2], deap_obs[2], axis=0)]
            priors      = np.append(priors, deap_priors, axis=0)
            #priors      = np.append(priors, np.zeros((deap_actions.shape[0], priors.shape[1], priors.shape[2]), dtype=np.int32), axis=0)

            
        # Retrieve metrics
        base_r      = np.array([p.base_r for p in programs])
        r           = np.array([p.r for p in programs])
        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        on_policy   = np.array([p.on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)
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
        max_item = np.argmax(base_r)
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
        
        '''
            Risk-seeking policy gradient: only train on top epsilon fraction of sampled expressions
            Note: controller.train_step(r_train, b_train, actions, obs, priors, mask, priority_queue)
        
            GP Integration note:
            
            For the moment, GP samples get added on top of the epsilon samples making it slightly larger. 
            This will be changed in the future when we integrate off policy support.
        '''
        if epsilon is not None and epsilon < 1.0:
            n_keep      = int(epsilon * batch_size) # Number of top indices to keep
            keep        = np.zeros(shape=(base_r.shape[0],), dtype=bool)
            keep[np.argsort(r)[-n_keep:]] = True
            
            # These guys can contain the GP solutions if we run GP
            '''
                Here we get the returned as well as stored programs and properties. 
                
                If we are returning the GP programs to the controller, p and r will be exactly the same
                as p_train and r_train. Othewrwise, p and r will still contain the GP programs so they
                can still fall into the hall of fame. p_train and r_train will be different and no longer
                contain the GP program items. 
            '''

            base_r      = base_r[keep]
            l           = l[keep]
            s           = list(compress(s, keep))
            invalid     = invalid[keep]
            
            # Option: don't keep the GP programs for return to controller
            if run_gp_meld and not gp_controller.return_gp_obs:
                if gp_verbose:
                    print("GP solutions NOT returned to controller")
                '''
                    If we are not returning the GP components to the controller, we will remove them from
                    r_train and p_train by augmenting 'keep'. We just chop off the GP elements which are indexed
                    from batch_size to the end of the list.
                '''
                _r                  = r[keep]
                _p                  = list(compress(programs, keep))
                    
                keep[batch_size:]   = False
                
                r_train             = r[keep]
                p_train             = list(compress(programs, keep))
                
                '''
                    These contain all the programs and rewards regardless of whether they are returned to the controller.
                    This way, they can still be stored in the hall of fame. 
                '''
                r                   = _r
                programs            = _p
            else:
                if run_gp_meld and gp_verbose:
                    print("{} GP solutions returned to controller".format(gp_controller.config_gp_meld["train_n"]))
                '''
                    Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
                '''
                r_train = r         = r[keep]
                p_train = programs  = list(compress(programs, keep))            
                
            '''
                get the action, observation, priors and on_policy status of all programs returned to the controller.
            '''
            actions     = actions[keep, :]
            obs         = [o[keep, :] for o in obs]
            priors      = priors[keep, :, :]
            on_policy   = on_policy[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r       = np.clip(r,        -1e6, 1e6)
        r_train = np.clip(r_train,  -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if baseline == "ewma_R":
            ewma = np.mean(r_train) if ewma is None else alpha*np.mean(r_train) + (1 - alpha)*ewma
            b_train = ewma
        elif baseline == "R_e": # Default
            ewma = -1
            b_train = np.min(r_train) # The worst of the lot
        elif baseline == "ewma_R_e":
            ewma = np.min(r_train) if ewma is None else alpha*np.min(r_train) + (1 - alpha)*ewma
            b_train = ewma
        elif baseline == "combined":
            ewma = np.mean(r_train) - np.min(r_train) if ewma is None else alpha*(np.mean(r_train) - np.min(r_train)) + (1 - alpha)*ewma
            b_train = np.min(r_train) + ewma

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

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), controller.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r_train)

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.update(programs, sampled_batch)
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

        if gp_verbose:
            print("************************")
            print("Best step Program:")
            programs[np.argmax(base_r)].print_stats()
            print("************************")
            print("All time best Program:")
            p_base_r_best.print_stats()
            print("************************")

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
            
        if run_gp_meld:
            if nevals > n_samples:
                print("************************")
                print("All time best Program:")
                p_base_r_best.print_stats()
                print("************************")
                print("Max Number of Samples Exceeded. Exiting...")
                break

    if save_all_r:
        with open(all_r_output_file, 'ab') as f:
            np.save(f, all_r)

    
    # Save the hall of fame
    
    if hof is not None and hof > 0:
        # For stochastic Tasks, average each unique Program's base_r_history,
        if Program.task.stochastic:

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

            """
            NOTE: Equivalence class computation is too expensive.
            Refactor later based on reward and/or semantics.
            """
            # # Filter out symbolically unique Programs. Assume N/A expressions are unique.
            # if pool is not None:
            #     results = pool.map(sympy_work, programs)
            #     for p, result in zip(programs, results):
            #         p.sympy_expr = result[0]
            # else:
            #     results = list(map(sympy_work, programs))
            # str_sympy_exprs = [result[1] for result in results]
            # unique_ids = np.unique(str_sympy_exprs, return_index=True)[1].tolist()
            # na_ids = [i for i in range(len(str_sympy_exprs)) if str_sympy_exprs[i] == "N/A"]
            # programs = list(map(programs.__getitem__, unique_ids + na_ids))

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
        pf.sort(key=lambda p : p.complexity_eureqa) # Sort by complexity

        if pool is not None:
            results = pool.map(pf_work, pf)
        else:
            results = list(map(pf_work, pf))

        eval_keys = list(results[0][-1].keys())
        columns = ["complexity", "r", "base_r", "count", "expression", "traversal"] + eval_keys
        pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(pf_results, columns=columns)
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
        "traversal" : repr(p)
        })
    
    if output_file is not None:
        print("Results saved to: {}".format(output_file))
    
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
    task = make_task(**config_task)
    Program.set_task(task)
    Program.set_library(task.function_set, task.n_input_var)
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
