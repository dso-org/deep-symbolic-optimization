"""Utility functions used in deep symbolic regression."""

import os
import heapq
import functools
import numpy as np
from collections import namedtuple


Batch = namedtuple(
    "Batch", ["actions", "obs", "priors", "lengths", "rewards", "on_policy"])


def is_float(s):
    """Determine whether str can be cast to float."""

    try:
        float(s)
        return True
    except ValueError:
        return False


# Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def setup_output_files(logdir, output_file):
    """
    Writes the main output file header and returns the reward, hall of fame, and Pareto front config filenames.

    Parameters:
    -----------

    logdir : string
        Directory to log to.

    output_file : string
        Name of output file.

    Returns:
    --------

    all_r_output_file : string
        all_r output filename
    
    hof_output_file : string
        hof output filename

    pf_output_file : string
        pf output filename
    """
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

    return all_r_output_file, hof_output_file, pf_output_file

class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


# Adapted from https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/brain_coder/common/utils.py
class MPQItemContainer(object):
    """Class for holding an item with its score.

    Defines a comparison function for use in the heap-queue.
    """

    def __init__(self, score, item, extra_data):
        self.item = item
        self.score = score
        self.extra_data = extra_data


    def __lt__(self, other):
        assert isinstance(other, type(self))
        return self.score < other.score

    def __iter__(self):
        """Allows unpacking like a tuple."""
        yield self.score
        yield self.item
        yield self.extra_data

    def __repr__(self):
        """String representation of this item.

        `extra_data` is not included in the representation. We are assuming that
        `extra_data` is not easily interpreted by a human (if it was, it should be
        hashable, like a string or tuple).

        Returns:
            String representation of `self`.
        """
        return str((self.score, self.item))

    def __str__(self):
        return repr(self)


# Adapted from https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/brain_coder/common/utils.py
class MaxUniquePriorityQueue(object):
    """A maximum priority queue where duplicates are not added.

    The top items by score remain in the queue. When the capacity is reached,
    the lowest scored item in the queue will be dropped.

    This implementation differs from a typical priority queue, in that the minimum
    score is popped, instead of the maximum. Largest scores remain stuck in the
    queue. This is useful for accumulating the best known items from a population.

    The items used to determine uniqueness must be hashable, but additional
    non-hashable data may be stored with each item.
    """

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.heap = []
        self.unique_items = set()
        self.rng = np.random.RandomState(seed)

    def push(self, score, item, extra_data=None):
        """Push an item onto the queue.

        If the queue is at capacity, the item with the smallest score will be
        dropped. Note that it is assumed each item has exactly one score. The same
        item with a different score will still be dropped.

        Args:
            score: Number used to prioritize items in the queue. Largest scores are
                    kept in the queue.
            item: A hashable item to be stored. Duplicates of this item will not be
                    added to the queue.
            extra_data: An extra (possible not hashable) data to store with the item.
        """
        if item in self.unique_items:
            return
        if len(self.heap) >= self.capacity:
            _, popped_item, _ = heapq.heappushpop(
                    self.heap, MPQItemContainer(score, item, extra_data))
            self.unique_items.add(item)
            self.unique_items.remove(popped_item)
        else:
            heapq.heappush(self.heap, MPQItemContainer(score, item, extra_data))
            self.unique_items.add(item)

    def pop(self):
        """Pop the item with the lowest score.

        Returns:
            score: Item's score.
            item: The item that was popped.
            extra_data: Any extra data stored with the item.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.heappop(self.heap)
        self.unique_items.remove(item)
        return score, item, extra_data

    def get_max(self):
        """Peek at the item with the highest score.

        Returns:
            Same as `pop`.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.nlargest(1, self.heap)[0]
        return score, item, extra_data

    def get_min(self):
        """Peek at the item with the lowest score.

        Returns:
            Same as `pop`.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.nsmallest(1, self.heap)[0]
        return score, item, extra_data

    def random_sample(self, sample_size):
        """Randomly select items from the queue.

        This does not modify the queue.

        Items are drawn from a uniform distribution, and not weighted by score.

        Args:
            sample_size: Number of random samples to draw. The same item can be
                    sampled multiple times.

        Returns:
            List of sampled items (of length `sample_size`). Each element in the list
            is a tuple: (item, extra_data).
        """
        idx = self.rng.choice(len(self.heap), sample_size, )
        return [(self.heap[i].item, self.heap[i].extra_data) for i in idx]

    def iter_in_order(self):
        """Iterate over items in the queue from largest score to smallest.

        Yields:
            item: Hashable item.
            extra_data: Extra data stored with the item.
        """
        for _, item, extra_data in heapq.nlargest(len(self.heap), self.heap):
            yield item, extra_data

    def sample_batch(self, sample_size):
        """Randomly select items from the queue and return them as a Batch."""

        dicts = [extra_data for (item, extra_data) in self.random_sample(sample_size)]
        actions = np.stack([d["actions"] for d in dicts], axis=0)
        obs = tuple([np.stack([d["obs"][i] for d in dicts], axis=0) for i in range(3)])
        priors = np.stack([d["priors"] for d in dicts], axis=0)
        lengths = np.array([d["lengths"] for d in dicts], dtype=np.int32)
        rewards = np.array([d["program"].r for d in dicts], dtype=np.float32)
        batch = Batch(actions=actions, obs=obs, priors=priors, lengths=lengths,
                      rewards=rewards)
        return batch

    def update(self, programs, batch):
        """
        Update the queue with the single best Program in a batch of Programs.

        Parameters
        ----------

        programs : list of Program
            List of Programs.

        batch : Batch
            Batch data corresponding to Programs.
        """

        i = np.argmax(batch.rewards)
        p = programs[i]
        score = p.r
        item = p.tokens.tostring()
        extra_data = {
            "actions" : batch.actions[i],
            "obs" : [o[i] for o in batch.obs],
            "priors" : batch.priors[i],
            "lengths" : batch.lengths[i],
            "program" : p
        }
        self.push(score, item, extra_data)

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        for _, item, _ in self.heap:
            yield item

    def __repr__(self):
        return '[' + ', '.join(repr(c) for c in self.heap) + ']'

    def __str__(self):
        return repr(self)


# Entropy computation in batch
def empirical_entropy(labels):

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return ent
