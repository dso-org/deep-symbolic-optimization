import multiprocessing

import numpy as np
from pathos.multiprocessing import ProcessPool
from deap import gp
from deap import base
from deap import tools
from deap import creator

from dso.subroutines import jit_parents_siblings_at_once
from dso.gp import base as gp_base
from dso.program import Program, from_tokens
import dso.gp.utils as U


class GPController:

    def __init__(self, prior, verbose=False, generations=20, p_crossover=0.5,
                 p_mutate=0.5, tournament_size=3, mutate_tree_max=2, train_n=50,
                 parallel_eval=False):
        """
        Parameters
        ----------
        prior : dso.prior.JointPrior
            JointPrior used to determine whether constraints were violated.

        verbose : bool
            Whether to print GP diagnostics.

        generations : int
            Number of GP generations between each RL step.

        p_crossover : float
            Probability that a pair of individuals will undergo crossover.

        p_mutate : float
            Probability that an individual will undergo mutation.

        tournament_size : int
            Tournament size used for selection by tools.selTournament.

        mutate_tree_max : int
            Maximum tree depth for mutation insertions.

        train_n : int
            Number of GP individuals to return to RL.

        parallel_eval : bool
            Use multiprocessing to evaluate individuals?
        """

        self.prior = prior
        self.pset = U.create_primitive_set(Program.library)
        self.verbose = verbose
        self.generations = generations
        self.train_n = train_n
        self.return_gp_obs = self.train_n > 0
        self.nevals = 0

        # Create a Hall of Fame object
        if self.train_n > 0:
            self.hof = tools.HallOfFame(maxsize=self.train_n)

        # Create a DEAP toolbox
        self.toolbox, self.creator = self._create_toolbox(self.pset,
                                                          parallel_eval=parallel_eval,
                                                          tournament_size=tournament_size,
                                                          mutate_tree_max=mutate_tree_max)

        # Actual loop function that runs GP
        self.algorithm = gp_base.RunOneStepAlgorithm(toolbox=self.toolbox,
                                                     cxpb=p_crossover,
                                                     mutpb=p_mutate,
                                                     verbose=verbose)

    def check_constraint(self, individual):
        actions, parents, siblings = U.individual_to_dso_aps(individual, Program.library)
        return self.prior.is_violated(actions, parents, siblings)

    def _create_toolbox(self, pset, tournament_size=3, mutate_tree_max=5,
                        parallel_eval=True):
        """
        Create a deap.base.Toolbox.
        """

        assert isinstance(pset, gp.PrimitiveSet), "pset should be a PrimitiveSet"

        # NOTE from deap.creator.create: create(name, base, **kargs):
        # ALSO: Creates a new class named *name* inheriting from *base*

        # Create custom fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Adds fitness into PrimitiveTree

        # NOTE from deap.base.Toolbox:  def register(self, alias, function, *args, **kargs):
        # ALSO the function in toolbox is defined as: partial(function, *args, **kargs)

        # Define the evolutionary operators
        toolbox = base.Toolbox()
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=mutate_tree_max)
        toolbox.register('mutate', U.multi_mutate, expr=toolbox.expr_mut, pset=pset)

        # Decorate mate and mutate by checking for constraint violation
        toolbox.decorate("mate", U.staticLimit(key=self.check_constraint, max_value=0))
        toolbox.decorate("mutate", U.staticLimit(key=self.check_constraint, max_value=0))

        # Overide the built-in map function
        if parallel_eval:
            print("GP Controller using parallel evaluation via Pathos")
            pool = ProcessPool(nodes=multiprocessing.cpu_count())
            print("\t>>> Using {} processes".format(pool.ncpus))
            toolbox.register("cmap", pool.map)
        else:
            toolbox.register("cmap", map)

        # Create the training function
        return toolbox, creator

    def get_hof_programs(self):
        """Compute actions, parents, siblings, and priors of hall of fame."""

        hof = self.hof
        L = Program.library.L

        actions = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_action = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_parent = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_sibling = np.zeros((len(hof), self.max_length), dtype=np.int32)

        obs_action[:, 0] = L # TBD: EMPTY_ACTION
        programs = []

        # TBD: Utility function to go from actions -> (obs_actions, obs_parent, obs_sibling)
        # (Similar to test_prior.py:make_batch)

        # Compute actions, obs (action, parent, sibling), and programs
        for i, ind in enumerate(hof):
            tokens = U.DEAP_to_padded_tokens(ind, self.max_length)
            actions[i, :] = tokens
            obs_action[i, 1:] = tokens[:-1]
            obs_parent[i, :], obs_sibling[i, :] = jit_parents_siblings_at_once(np.expand_dims(tokens, axis=0),
                                                                               arities=Program.library.arities,
                                                                               parent_adjust=Program.library.parent_adjust)
            programs.append(from_tokens(tokens, optimize=False, on_policy=False))

        # Compute priors
        if self.train_n > 0:
            priors = self.prior.at_once(actions, obs_parent, obs_sibling)
        else:
            priors = np.zeros((len(programs), self.max_length, L), dtype=np.float32)

        obs = (obs_action, obs_parent, obs_sibling)

        return programs, actions, obs, priors

    def __call__(self, actions):
        """
        Parameters
        ----------

        actions : np.ndarray
            Actions to use as starting population.
        """

        # TBD: Fix hack
        self.max_length = actions.shape[1]

        # Reset the HOF
        if self.hof is not None:
            self.hof = tools.HallOfFame(maxsize=self.train_n)

        # Get DSO generated batch members into Deap based "individuals"
        # TBD: Can base class of Individual can be initialized with tokens and Program?
        individuals = [self.creator.Individual(U.tokens_to_DEAP(a, self.pset)) for a in actions]
        self.algorithm.set_population(individuals)

        # Run GP generations
        self.nevals = 0
        for i in range(self.generations):
            nevals = self.algorithm(self.hof) # Run one generation
            self.nevals += nevals

        # Get the HOF batch
        if self.train_n > 0:
            programs, actions, obs, priors = self.get_hof_programs()

        return programs, actions, obs, priors

    def __del__(self):
        del self.creator.FitnessMin
