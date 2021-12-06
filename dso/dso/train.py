"""Defines main training loop for deep symbolic optimization."""

import os
import time
from itertools import compress

import tensorflow as tf
import numpy as np

from dso.program import Program, from_tokens
from dso.utils import empirical_entropy, get_duration, weighted_quantile
from dso.memory import Batch, make_queue
from dso.variance import quantile_variance
from dso.train_stats import StatsLogger

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.set_random_seed(0)


# Work for multiprocessing pool: compute reward
def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r
    return p


def learn(sess, controller, pool, gp_controller, output_file,
          n_epochs=None, n_samples=2000000, batch_size=1000, complexity="token",
          const_optimizer="scipy", const_params=None, alpha=0.5,
          epsilon=0.05, n_cores_batch=1, verbose=True, save_summary=False,
          save_all_epoch=False, baseline="R_e",
          b_jumpstart=False, early_stopping=True, hof=100, eval_all=False,
          save_pareto_front=True, debug=0, use_memory=False, memory_capacity=1e3,
          warm_start=None, memory_threshold=None, save_positional_entropy=False,
          save_top_samples_per_batch=0, save_cache=False,
          save_cache_r_min=0.9, save_freq=None, save_token_count=False):

    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TensorFlow Session object.

    controller : dso.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

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

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
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

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """

    run_gp_meld = gp_controller is not None

    # Config assertions and warnings
    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    # Initialize compute graph
    sess.run(tf.global_variables_initializer())

    if debug:
        tvars = tf.trainable_variables()

        def print_var_means():
            tvars_vals = sess.run(tvars)
            for var, val in zip(tvars, tvars_vals):
                print(var.name, "mean:", val.mean(), "var:", val.var())

    # Create the priority queue
    k = controller.pqt_k
    if controller.pqt and k is not None and k > 0:
        priority_queue = make_queue(priority=True, capacity=k)
    else:
        priority_queue = None

    # Create the memory queue
    if use_memory:
        assert epsilon is not None and epsilon < 1.0, \
            "Memory queue is only used with risk-seeking."
        memory_queue = make_queue(controller=controller, priority=False,
                                  capacity=int(memory_capacity))

        # Warm start the queue
        warm_start = warm_start if warm_start is not None else batch_size
        actions, obs, priors = controller.sample(warm_start)
        programs = [from_tokens(a) for a in actions]
        r = np.array([p.r for p in programs])
        l = np.array([len(p.traversal) for p in programs])
        on_policy = np.array([p.originally_on_policy for p in programs])
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=l, rewards=r, on_policy=on_policy)
        memory_queue.push_batch(sampled_batch, programs)
    else:
        memory_queue = None

    if debug >= 1:
        print("\nInitial parameter means:")
        print_var_means()

    # For stochastic Tasks, store each reward computation for each unique traversal
    if Program.task.stochastic:
        r_history = {} # Dict from Program str to list of rewards
        # It's not really clear whether Programs with const should enter the hof for stochastic Tasks
        assert Program.library.const_token is None, \
            "Constant tokens not yet supported with stochastic Tasks."
        assert not save_pareto_front, "Pareto front not supported with stochastic Tasks."
    else:
        r_history = None

    # Main training loop
    p_final = None
    r_best = -np.inf
    prev_r_best = None
    ewma = None if b_jumpstart else 0.0 # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    nevals = 0 # Total number of sampled expressions (from RL or GP)
    positional_entropy = np.zeros(shape=(n_epochs, controller.max_length), dtype=np.float32)

    top_samples_per_batch = list()

    logger = StatsLogger(sess, output_file, save_summary, save_all_epoch, hof, save_pareto_front,
                         save_positional_entropy, save_top_samples_per_batch, save_cache,
                         save_cache_r_min, save_freq, save_token_count)

    start_time = time.time()
    if verbose:
        print("-- RUNNING EPOCHS START -------------")
    for epoch in range(n_epochs):

        # Set of str representations for all Programs ever seen
        s_history = set(r_history.keys() if Program.task.stochastic else Program.cache.keys())

        # Sample batch of Programs from the Controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors = controller.sample(batch_size)
        programs = [from_tokens(a) for a in actions]
        nevals += batch_size

        # Run GP seeded with the current batch, returning elite samples
        if run_gp_meld:
            deap_programs, deap_actions, deap_obs, deap_priors = gp_controller(actions)
            nevals += gp_controller.nevals

            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)

        # Compute rewards in parallel
        if pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))          
            pool_p_dict = { p.str : p for p in pool.map(work, programs_to_optimize) }   
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__  else p for p in programs]
            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)
                    
        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])
        r_train = r

        # Back up programs to save them properly later
        controller_programs = programs.copy() if save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train     = programs

        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        on_policy   = np.array([p.originally_on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)

        if save_positional_entropy:
            positional_entropy[epoch] = np.apply_along_axis(empirical_entropy, 0, actions)

        if save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            one_perc = int(len(programs) * float(save_top_samples_per_batch))
            for idx in sorted_idx[:one_perc]:
                top_samples_per_batch.append([epoch, r[idx], repr(programs[idx])])

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]
            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]

        # Update reward history
        if r_history is not None:
            for p in programs:
                key = p.str
                if key in r_history:
                    r_history[key].append(p.r)
                else:
                    r_history[key] = [p.r]

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)
        r_best = max(r_max, r_best)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if epsilon is not None and epsilon < 1.0:
            # Compute reward quantile estimate
            if use_memory: # Memory-augmented quantile
                # Get subset of Programs not in buffer
                unique_programs = [p for p in programs \
                                   if p.str not in memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = memory_queue.get_rewards()
                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum() # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > memory_threshold:
                        quantile_variance(memory_queue, controller, batch_size, epsilon, epoch)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - epsilon)

            else: # Empirical quantile
                quantile = np.quantile(r, 1 - epsilon, interpolation="higher")

            # These guys can contain the GP solutions if we run GP
            '''
                Here we get the returned as well as stored programs and properties.

                If we are returning the GP programs to the controller, p and r will be exactly the same
                as p_train and r_train. Othewrwise, p and r will still contain the GP programs so they
                can still fall into the hall of fame. p_train and r_train will be different and no longer
                contain the GP program items.
            '''

            keep        = r >= quantile
            l           = l[keep]
            s           = list(compress(s, keep))
            invalid     = invalid[keep]

            # Option: don't keep the GP programs for return to controller
            if run_gp_meld and not gp_controller.return_gp_obs:
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
                '''
                    Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
                '''
                r_train = r         = r[keep]
                p_train = programs  = list(compress(programs, keep))

            '''
                get the action, observation, priors and on_policy status of all programs returned to the controller.
            '''
            actions     = actions[keep, :]
            obs         = obs[keep, :, :]
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
            b_train = quantile
        elif baseline == "ewma_R_e":
            ewma = np.min(r_train) if ewma is None else alpha*quantile + (1 - alpha)*ewma
            b_train = ewma
        elif baseline == "combined":
            ewma = np.mean(r_train) - quantile if ewma is None else alpha*(np.mean(r_train) - quantile) + (1 - alpha)*ewma
            b_train = quantile + ewma

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), controller.max_length)
                            for p in p_train], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r_train, on_policy=on_policy)

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, programs)
            pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
        else:
            pqt_batch = None

        # Train the controller
        summaries = controller.train_step(b_train, sampled_batch, pqt_batch)

        #wall time calculation for the epoch
        epoch_walltime = time.time() - start_time

        # Collect sub-batch statistics and write output
        logger.save_stats(r_full, l_full, actions_full, s_full, invalid_full, r,
                          l, actions, s, invalid, r_best, r_max, ewma, summaries, epoch,
                          s_history, b_train, epoch_walltime, controller_programs)

        # Update the memory queue
        if memory_queue is not None:
            memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        new_r_best = False

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = programs[np.argmax(r)]

        prev_r_best = r_best

        # Print new best expression
        if verbose and new_r_best:
            print("[{}] Training epoch {}/{}, current best R: {:.4f}".format(get_duration(start_time), epoch + 1, n_epochs, prev_r_best))
            print("\n\t** New best")
            p_r_best.print_stats()

        # Stop if early stopping criteria is met
        if eval_all and any(success):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break
        if early_stopping and p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            break

        if verbose and (epoch + 1) % 10 == 0:
            print("[{}] Training epoch {}/{}, current best R: {:.4f}".format(get_duration(start_time), epoch + 1, n_epochs, prev_r_best))

        if debug >= 2:
            print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            print_var_means()

        if verbose and (epoch + 1) == n_epochs:
            print("[{}] Ending training after epoch {}/{}, current best R: {:.4f}".format(get_duration(start_time), epoch + 1, n_epochs, prev_r_best))

        if nevals > n_samples:
            break

    if verbose:
        print("-- RUNNING EPOCHS END ---------------\n")
        print("-- EVALUATION START ----------------")
        #print("\n[{}] Evaluating the hall of fame...\n".format(get_duration(start_time)))

    controller.prior.report_constraint_counts()

    #Save all results available only after all epochs are finished. Also return metrics to be added to the summary file
    results_add = logger.save_results(positional_entropy, top_samples_per_batch, r_history, pool, epoch, nevals)

    # Print the priority queue at the end of training
    if verbose and priority_queue is not None:
        for i, item in enumerate(priority_queue.iter_in_order()):
            print("\nPriority queue entry {}:".format(i))
            p = Program.cache[item[0]]
            p.print_stats()

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_r_best
    result = {
        "r" : p.r,
    }
    result.update(p.evaluate)
    result.update({
        "expression" : repr(p.sympy_expr),
        "traversal" : repr(p),
        "program" : p
        })
    result.update(results_add)

    if verbose:
        print("-- EVALUATION END ------------------")
    return result
