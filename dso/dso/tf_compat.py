"""TensorFlow 2.x compatibility module for DSO.

This module provides centralized TF2 imports and utility functions
for GPU memory configuration and seeding.
"""

import tensorflow as tf


def configure_gpu_memory():
    """Configure GPU memory growth to avoid allocating all GPU memory at once.

    This is the TF2 equivalent of setting allow_growth in session config.
    Should be called early in program initialization.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU memory growth configuration failed: {e}")


def set_seeds(seed):
    """Set random seeds for TensorFlow and reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    tf.random.set_seed(seed)


def limit_parallelism(intra_threads=1, inter_threads=1):
    """Limit TensorFlow parallelism to prevent resource contention.

    This is useful when running multiple DSO instances in parallel.

    Parameters
    ----------
    intra_threads : int
        Number of threads for intra-op parallelism.
    inter_threads : int
        Number of threads for inter-op parallelism.
    """
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
