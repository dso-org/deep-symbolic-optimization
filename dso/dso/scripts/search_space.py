"""
Tools for Monte Carlo estimates of search space reduction. Search space
reduction "from A to B" is estimated by generating samples from configuration
A, then computing the fraction of samples that are constrained under
configuration B.
"""

from copy import deepcopy

import tensorflow as tf
import numpy as np
import click

from dso import DeepSymbolicOptimizer
from dso.program import Program, from_tokens


def create_model(config, prior_override=None):
    model = DeepSymbolicOptimizer(config)
    if prior_override is not None:
        model.config["prior"] = prior_override
        model.config_prior = prior_override
    model.setup()
    model.sess.run(tf.global_variables_initializer())

    return model


def count_violations(actions, obs, model):
    """
    Given candidate actions, count the number of sequences that are constrained
    under model.
    """

    # Compute sequence lengths
    programs = [from_tokens(a) for a in actions]
    lengths = np.array([min(len(p.traversal), model.policy.max_length) for p in programs], dtype=np.int32)

    # Compute priors under model
    priors = model.prior.at_once(actions, obs[:, 1, :], obs[:, 2, :])

    # Count constraint violations under model
    count = 0
    T = actions.shape[1]
    for i in range(actions.shape[0]):
        count += (priors[i, np.arange(T), actions[i]][:lengths[i]] == -np.inf).any()

    return count


@click.command()
@click.argument("config1", required=False, default=None, type=str)
@click.argument("config2", required=False, default=None, type=str)
@click.option("--n", type=int, default=1000, help="Number of samples to generate.")
@click.option("--mode", type=click.Choice(["marginal", "single", "all", None]), default=None)
def main(config1, config2, n, mode):

    """
    For each prior P, estimate search space reduction from all-but-P to all
    priors.
    """
    if mode == "marginal":
        assert config2 is None, "'Marginal' mode works with a single config."

        # Create the original model
        model1 = create_model(config1)

        # For each Prior, compute search space reduction with that prior "off"
        counts = {}
        for k, v in model1.config["prior"].items():

            # Skip non-dict values and disabled priors
            if not isinstance(v, dict) or not v.get("on", False):
                continue

            # Disable the prior
            config2 = deepcopy(model1.config)
            config2["prior"][k]["on"] = False

            # Sample from config2
            model2 = create_model(config2)
            actions, obs, _ = model2.policy.sample(n)

            # Count violations under model1
            counts[k] = count_violations(actions, obs, model1)

        for k, v in counts.items():
            print("Full prior constrained {}/{} samples from all-but-{}.".format(v, n, k))

    """
    For each prior P, estimate search space reduction from no priors to P.
    """
    if mode == "single":
        assert config2 is None, "'Single' mode works with a single config."

        # Get the config with all priors
        config_all = create_model(config1).config

        # Sample from the no-prior model
        model_none = create_model(config1, prior_override={})
        actions, obs, _ = model_none.policy.sample(n)

        # For each Prior, compute search space reduction with that prior "on"
        counts = {}
        for k, v in config_all["prior"].items():

            # Skip non-dict values and disabled priors
            if not isinstance(v, dict) or not v.get("on", False):
                continue

            # Create a model with a single prior
            config2 = deepcopy(config_all)
            model2 = create_model(config2, prior_override={k : v})

            # Count constraints
            counts[k] = count_violations(actions, obs, model2)

        for k, v in counts.items():
            print("Prior '{}' alone constrained {}/{} samples from no-prior config.".format(k, v, n))

    """
    Estimate search space reduction from no priors to config1.
    """
    if mode == "all":
        assert config2 is None, "'All' mode works with a single config."

        # Sample from the no-prior model
        model_none = create_model(config1, prior_override={})
        actions, obs, _ = model_none.policy.sample(n)

        # Compare to full prior
        model2 = create_model(config1)
        count = count_violations(actions, obs, model2)

        print("The config constrained {}/{} samples from no-prior config.".format(count, n))

    """
    Estimate search space reduction from config1 to config2.
    """
    if mode is None:

        # Sample from config1
        model1 = create_model(config1)
        actions, obs, _ = model1.policy.sample(n)
        tokens1 = set(Program.library.names)

        # Count violations under model2
        model2 = create_model(config2)
        tokens2 = set(Program.library.names)
        assert tokens1 == tokens2, "Tokens must be the same between config1 and config2."
        count = count_violations(actions, obs, model2)

        print("The new config constrained {}/{} samples from the old config.".format(count, n))


if __name__ == "__main__":
    main()
