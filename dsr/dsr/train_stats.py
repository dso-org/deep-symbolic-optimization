"""Performs computations and file manipulations for train statistics logging purposes"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
from dsr.program import Program, from_tokens
from dsr.utils import is_pareto_efficient, empirical_entropy
from itertools import compress
from io import StringIO
import shutil
from collections import defaultdict

#These functions are defined globally so they are pickleable and can be used by Pool.map
def hof_work(p):
    return [p.r, p.on_policy_count, p.off_policy_count, repr(p.sympy_expr), repr(p), p.evaluate]

def pf_work(p):
    return [p.complexity, p.r, p.on_policy_count, p.off_policy_count, repr(p.sympy_expr), repr(p), p.evaluate]


class StatsLogger():
    """ Class responsible for dealing with output files of training statistics. It encapsulates all outputs to files."""

    def __init__(self, sess, logdir="./log", save_summary=True, output_file=None, save_all_epoch=False, hof=10,
                 save_pareto_front=False, save_positional_entropy=False, save_cache=False, save_cache_r_min=0.9,
                 save_freq = None):
        """"
        sess : tf.Session
            TenorFlow Session object (used for generating summary files)

        logdir : str, optional
            Name of log directory.

        save_summary : bool, optional
            Whether to write TensorFlow summaries.

        output_file : str, optional
            Filename to write results for each iteration.

        save_all_epoch : bool, optional
            Whether to save statistics for all programs for each iteration.

        hof : int or None, optional
            Number of top Programs to evaluate after training.

        save_pareto_front : bool, optional
            If True, compute and save the Pareto front at the end of training.

        save_positional_entropy : bool, optional
            Whether to save evolution of positional entropy for each iteration.

        save_cache : bool
            Whether to save the str, count, and r of each Program in the cache.

        save_cache_r_min : float or None
            If not None, only keep Programs with r >= r_min when saving cache.

        save_freq : int or None
            Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf
        """
        self.sess = sess
        self.logdir = logdir
        self.save_summary = save_summary
        self.output_file = output_file
        self.save_all_epoch = save_all_epoch
        self.hof = hof
        self.save_pareto_front = save_pareto_front
        self.save_positional_entropy = save_positional_entropy
        self.save_cache = save_cache
        self.save_cache_r_min = save_cache_r_min
        self.all_r = []   # save all R separately to keep backward compatibility with a generated file.

        if save_freq is None:
            self.buffer_frequency = 1
        elif save_freq < 1:
            self.buffer_frequency = float('inf')
        else:
            self.buffer_frequency = save_freq

        self.buffer_epoch_stats = StringIO() #Buffer for epoch statistics
        self.buffer_all_programs = StringIO()  #Buffer for the statistics for all programs.

        self.setup_output_files()

    def setup_output_files(self):
        """
        Opens and prepares all output log files controlled by this class.
        """
        if self.output_file is not None:
            os.makedirs(self.logdir, exist_ok=True)
            self.output_file = os.path.join(self.logdir, self.output_file)
            prefix, _ = os.path.splitext(self.output_file)
            self.all_r_output_file = "{}_all_r.npy".format(prefix)
            self.all_info_output_file = "{}_all_info.csv".format(prefix)
            self.hof_output_file = "{}_hof.csv".format(prefix)
            self.pf_output_file = "{}_pf.csv".format(prefix)
            self.positional_entropy_output_file = "{}_positional_entropy.npy".format(prefix)
            self.cache_output_file = "{}_cache.csv".format(prefix)
            with open(self.output_file, 'w') as f:
                # r_best : Maximum across all iterations so far
                # r_max : Maximum across this iteration's batch
                # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
                # r_avg_sub : Average across this iteration's epsilon-subset batch
                # n_unique_* : Number of unique Programs in batch
                # n_novel_* : Number of never-before-seen Programs per batch
                # a_ent_* : Empirical positional entropy across sequences averaged over positions
                # invalid_avg_* : Fraction of invalid Programs per batch
                # baseline: Baseline value used for training
                # time: time used to learn in this epoch (in seconds)
                headers = ["r_best",
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
                           "invalid_avg_sub",
                           "baseline",
                           "time"]
                f.write("{}\n".format(",".join(headers)))

                if self.save_all_epoch:
                    with open(self.all_info_output_file, 'w') as f:
                        # epoch : The epoch in which this line was saved
                        # r : reward for this program
                        # l : length of the program
                        # invalid : if the program is invalid
                        headers = ["epoch",
                                    "r",
                                    "l",
                                    "invalid"]
                        f.write("{}\n".format(",".join(headers)))
        else:
            self.all_r_output_file = self.hof_output_file = self.pf_output_file = self.positional_entropy_output_file = \
                self.cache_output_file = self.all_info_output_file = None
        # Creates the summary writer
        if self.save_summary:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            if self.output_file is not None:
                summary_dir = os.path.join(self.logdir, "summary_" + timestamp)
            else:
                summary_dir = os.path.join("summary", timestamp)
            self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        else:
            self.summary_writer = None

    def save_stats(self, r_full, l_full, actions_full, s_full, invalid_full, r, l,
                   actions, s, invalid, r_best, r_max, ewma, summaries, epoch, s_history,
                   baseline, epoch_walltime):
        """
        Computes and saves all statistics that are computed for every time step. Depending on the value of
            self.buffer_frequency, the statistics might be instead saved in a buffer before going to disk.
        :param r_full: The reward of all programs
        :param l_full: The length of all programs
        :param actions_full: all actions sampled this step
        :param s_full: String representation of all programs sampled this step.
        :param invalid_full: boolean for all programs sampled showing if they are invalid
        :param r: r_full excluding the ones where keep=false
        :param l: l_full excluding the ones where keep=false
        :param actions: actions_full excluding the ones where keep=false
        :param s: s_full excluding the ones where keep=false
        :param invalid: invalid_full excluding the ones where keep=false
        :param r_best: reward from the all time best program so far
        :param r_max: reward from the best program in this epoch
        :param ewma: Exponentially Weighted Moving Average weight that might be used for baseline computation
        :param summaries: Sumarries returned by the Controller this step
        :param epoch: This epoch id
        :param s_history: all programs ever seen in string format.
        :param baseline: baseline value used for training
        :param epoch_walltime: time taken to process this epoch
        """
        epoch = epoch + 1 #changing from 0-based index to 1-based
        if self.output_file is not None:
            r_avg_full = np.mean(r_full)

            l_avg_full = np.mean(l_full)
            a_ent_full = np.mean(np.apply_along_axis(empirical_entropy, 0, actions_full))
            n_unique_full = len(set(s_full))
            n_novel_full = len(set(s_full).difference(s_history))
            invalid_avg_full = np.mean(invalid_full)

            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            a_ent_sub = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
            n_unique_sub = len(set(s))
            n_novel_sub = len(set(s).difference(s_history))
            invalid_avg_sub = np.mean(invalid)
            stats = np.array([[
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
                invalid_avg_sub,
                baseline,
                epoch_walltime
            ]], dtype=np.float32)
            np.savetxt(self.buffer_epoch_stats, stats, delimiter=',')
        if self.save_all_epoch:
            all_epoch_stats = np.array([
                              [epoch]*len(r_full),
                              r_full,
                              l_full,
                              invalid_full
                              ]).transpose()
            df = pd.DataFrame(all_epoch_stats)
            df.to_csv(self.buffer_all_programs, mode='a', header=False, index=False, line_terminator='\n')

        # summary writers have their own buffer
        if self.save_summary:
            self.summary_writer.add_summary(summaries, epoch)

        # Should the buffer be saved now?
        if epoch % self.buffer_frequency == 0:
            if self.output_file is not None:
                self.flush_buffer(False)
            if self.save_all_epoch:
                self.flush_buffer(True)
            if self.summary_writer:
                self.summary_writer.flush()

        #Backwards compatibility of all_r numpy file
        if self.save_all_epoch:
            self.all_r.append(r_full)


    def save_results(self, positional_entropy, r_history, pool, n_epochs, n_samples):
        """
        Saves stats that are available only after all epochs are finished
        :param positional_entropy: evolution of positional_entropy for all epochs
        :param r_history: reward for each unique program found during training
        :param pool: Pool used to parallelize reward computation
        :param n_epochs: index of last epoch
        :param n_samples: Total number of samples
        """
        n_epochs = n_epochs + 1
        # First of all, saves any pending buffer
        if self.output_file is not None:
            self.flush_buffer(False)
        if self.save_all_epoch:
            self.flush_buffer(True)

        if self.summary_writer:
            self.summary_writer.flush()

        if self.save_all_epoch:
            #Kept all_r numpy file for backwards compatibility.
            with open(self.all_r_output_file, 'ab') as f:
                all_r = np.array(self.all_r, dtype=np.float32)
                np.save(f, all_r)

        if self.save_positional_entropy:
            with open(self.positional_entropy_output_file, 'ab') as f:
                np.save(f, positional_entropy)

        # Save the hall of fame
        if self.hof is not None and self.hof > 0:
            # For stochastic Tasks, average each unique Program's r_history,
            if Program.task.stochastic:

                # Define a helper function to generate a Program from its tostring() value
                def from_token_string(str_tokens, optimize):
                    tokens = np.fromstring(str_tokens, dtype=np.int32)
                    return from_tokens(tokens, optimize=optimize)

                # Generate each unique Program and manually set its reward to the average of its r_history
                keys = r_history.keys()  # str_tokens for each unique Program
                vals = r_history.values()  # reward histories for each unique Program
                programs = [from_token_string(str_tokens, optimize=False) for str_tokens in keys]
                for p, r in zip(programs, vals):
                    p.r = np.mean(r)
                    #It is not possible to tell if each program was sampled on- or off-policy at this point.
                    # -1 on off_policy_count signals that we can't distinguish the counters in this task.
                    p.on_policy_count = len(r)
                    p.off_policy_count = -1

            # For deterministic Programs, just use the cache
            else:
                programs = list(Program.cache.values())  # All unique Programs found during training

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


            r = [p.r for p in programs]
            i_hof = np.argsort(r)[-self.hof:][::-1]  # Indices of top hof Programs
            hof = [programs[i] for i in i_hof]


            if pool is not None:
                results = pool.map(hof_work, hof)
            else:
                results = list(map(hof_work, hof))

            eval_keys = list(results[0][-1].keys())
            columns = ["r", "count_on_policy", "count_off_policy", "expression", "traversal"] + eval_keys
            hof_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
            df = pd.DataFrame(hof_results, columns=columns)
            if self.hof_output_file is not None:
                print("Saving Hall of Fame to {}".format(self.hof_output_file))
                df.to_csv(self.hof_output_file, header=True, index=False)

            #save cache
            if self.save_cache and Program.cache:
                print("Saving cache to {}".format(self.cache_output_file))
                cache_data = [(repr(p), p.on_policy_count, p.off_policy_count, p.r) for p in Program.cache.values()]
                df_cache = pd.DataFrame(cache_data)
                df_cache.columns = ["str", "count_on_policy", "count_off_policy", "r"]
                if self.save_cache_r_min is not None:
                    df_cache = df_cache[df_cache["r"] >= self.save_cache_r_min]
                df_cache.to_csv(self.cache_output_file, header=True, index=False)

            # Compute the pareto front
            if self.save_pareto_front:
                #if verbose:
                #    print("Evaluating the pareto front...")
                all_programs = list(Program.cache.values())
                costs = np.array([(p.complexity, -p.r) for p in all_programs])
                pareto_efficient_mask = is_pareto_efficient(costs)  # List of bool
                pf = list(compress(all_programs, pareto_efficient_mask))
                pf.sort(key=lambda p: p.complexity)  # Sort by complexity

                if pool is not None:
                    results = pool.map(pf_work, pf)
                else:
                    results = list(map(pf_work, pf))

                eval_keys = list(results[0][-1].keys())
                columns = ["complexity", "r", "count_on_policy", "count_off_policy", "expression", "traversal"] + eval_keys
                pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
                df = pd.DataFrame(pf_results, columns=columns)
                if self.pf_output_file is not None:
                    print("Saving Pareto Front to {}".format(self.pf_output_file))
                    df.to_csv(self.pf_output_file, header=True, index=False)

                # Look for a success=True case within the Pareto front
                for p in pf:
                    if p.evaluate.get("success"):
                        p_final = p
                        break
            #Save error summaries
            # Print error statistics of the cache
            n_invalid = 0
            error_types = defaultdict(lambda: 0)
            error_nodes = defaultdict(lambda: 0)

            result = {}
            for p in Program.cache.values():
                if p.invalid:
                    count = p.off_policy_count + p.on_policy_count
                    n_invalid += count
                    error_types[p.error_type] += count
                    error_nodes[p.error_node] += count

            if n_invalid > 0:
                print("Invalid expressions: {} of {} ({:.1%}).".format(n_invalid, n_samples,
                                                                       n_invalid / n_samples))
                print("Error type counts:")
                for error_type, count in error_types.items():
                    print("  {}: {} ({:.1%})".format(error_type, count, count / n_invalid))
                    result["error_"+error_type] = count
                print("Error node counts:")
                for error_node, count in error_nodes.items():
                    print("  {}: {} ({:.1%})".format(error_node, count, count / n_invalid))
                    result["error_node_" + error_type] = count

            result['n_epochs'] = n_epochs
            result['n_samples'] = n_samples
            result['n_cached'] = len(Program.cache)
            return result
    def flush_buffer(self, all_info_buffer=False):
        """write buffer to output file
        @:param all_info_buffer: should self.buffer_epoch_stats (False) or self.buffer_all_programs (True) be flushed?

        """
        output = self.all_info_output_file if all_info_buffer else self.output_file
        buffer = self.buffer_all_programs if all_info_buffer else self.buffer_epoch_stats

        with open(output, 'a') as f:
            buffer.seek(0)
            shutil.copyfileobj(buffer, f, -1)

        # clear buffer
        if all_info_buffer:
            self.buffer_all_programs = StringIO()
        else:
            self.buffer_epoch_stats = StringIO()
