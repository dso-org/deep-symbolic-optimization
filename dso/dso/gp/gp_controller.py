import time

import numpy as np
from multiprocessing import Pool, cpu_count
from deap import gp
from deap import base
from deap import tools
from deap import creator

from dso.subroutines import jit_parents_siblings_at_once
from dso.gp import base as gp_base
from dso.program import Program, from_tokens
import dso.gp.utils as U


# library of crossover ops available to use
crossover_ops_dict = {
    "cxOnePoint" : U.cxOnePoint,
    "cxModifiedPMX" : U.cxModifiedPMX
}

# library of mutation ops available to use
mutation_ops_dict = {
    "multi_mutate" : U.multi_mutate,
    "multi_constrained_mutate" : U.multi_constrained_mutate
}


class GPController:

    def __init__(self, prior, config_prior, population_size=None, generations=20,
                 crossover_operator='cxOnePoint', p_crossover=0.5,
                 mutation_operator='multi_mutate', p_mutate=0.5, 
                 tournament_size=5, mutate_tree_max=3, train_n=50,
                 parallel_eval=False, ind_representation='', verbose=False,
                 **kwargs):
        """
        Parameters
        ----------
        prior : dso.prior.JointPrior
            JointPrior used to determine whether constraints were violated.

        config_prior : dict
            Priors config info. Needed to access constraint info in the priors.

        population_size : int or None
            Genetic Programming's population size. If None, then it is
            to DSO's batch size (all samples from current batch)

        generations : int
            Number of GP generations between each RL step.

        crossover_operator : str
            Crossover operator. It can be one of the available operators.

        p_crossover : float
            Probability that a pair of individuals will undergo crossover.

        mutation_operator : str
            Mutation operator. It can be one of the available operators.

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
        
        ind_representation : str
            Type of individual representation:
                'mod_diff': mod'ed difference from master sequence, or
                '': original representation (same as used by RL agent)

        verbose : bool
            Whether to print GP diagnostics.
        """

        self.prior = prior
        self.config_prior = config_prior
        self.population_size = population_size
        self.pset = U.create_primitive_set(Program.library)
        self.verbose = verbose
        self.generations = generations
        self.crossover_operator = crossover_operator
        self.p_crossover = p_crossover
        self.mutation_operator = mutation_operator
        self.p_mutate = p_mutate
        self.train_n = train_n
        self.nevals = 0
        self.total_nevals = 0
        self.ind_representation = ind_representation

        assert crossover_operator in crossover_ops_dict.keys(), \
                 'Crossover operator should be one of the following: ' \
                 '{}'.format([k for k in crossover_ops_dict.keys()])
        self.crossover_operator = crossover_operator

        assert mutation_operator in mutation_ops_dict.keys(), \
                'Mutation operator should be one of the following: ' \
                '{}'.format([k for k in mutation_ops_dict.keys()])
        self.mutation_operator = mutation_operator

        # Obtain constraint information from priors and task
        self.master_sequence = None
        self.max_mutations = None
        if Program.task.task_type == "binding":
            self.master_sequence = Program.task.master_actions
            if "mut_dist" in self.config_prior:
                self.max_mutations = self.config_prior['mut_dist']['max_']

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
                                                     p_crossover=p_crossover,
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
        creator.create("Individual", U.Individual,
                       fitness=creator.FitnessMin,  # Adds fitness into PrimitiveTree
                       num_mutations=0, 
                       max_mutations=self.max_mutations) 

        # NOTE from deap.base.Toolbox:  def register(self, alias, function, *args, **kargs):
        # ALSO the function in toolbox is defined as: partial(function, *args, **kargs)

        # Define the evolutionary operators
        toolbox = base.Toolbox()
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        cx_op = crossover_ops_dict[self.crossover_operator]
        toolbox.register("mate", cx_op) #, indpb=self.p_crossover)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=mutate_tree_max)
        mut_op = mutation_ops_dict[self.mutation_operator]
        toolbox.register('mutate', mut_op, expr=toolbox.expr_mut, pset=pset, indpb=self.p_mutate)

        # Decorate mate and mutate by checking for constraint violation
        toolbox.decorate("mate", U.staticLimit(key=self.check_constraint, max_value=0))
        toolbox.decorate("mutate", U.staticLimit(key=self.check_constraint, max_value=0))

        # Overide the built-in map function
        if parallel_eval:
            print("GP Controller using parallel evaluation")
            pool = Pool(cpu_count())
            print("\t>>> Using {} processes".format(cpu_count()))
            toolbox.register("cmap", pool.map)
        else:
            toolbox.register("cmap", map)

        # Create the training function
        return toolbox, creator

    def get_hof_programs(self):
        """Compute actions, parents, siblings, and priors of hall of fame."""
        hof = self.hof
        L = Program.library.L
        
        # Recheck maximum lengths
        self.max_length = max(max([len(ind) for i, ind in enumerate(hof)]), self.max_length)

        actions = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_action = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_parent = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_sibling = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_dangling = np.ones((len(hof), self.max_length), dtype=np.int32)

        obs_action[:, 0] = L # TBD: EMPTY_ACTION
        programs = []

        # TBD: Utility function to go from actions -> (obs_actions, obs_parent, obs_sibling)
        # (Similar to test_prior.py:make_batch)
        # Compute actions, obs, and programs
        for i, ind in enumerate(hof):

            ind.update_tree_repr()
            tokens = U.DEAP_to_padded_tokens(ind, self.max_length)

            actions[i, :] = tokens
            obs_action[i, 1:] = tokens[:-1]
            obs_parent[i, :], obs_sibling[i, :] = jit_parents_siblings_at_once(np.expand_dims(tokens, axis=0),
                                                                               arities=Program.library.arities,
                                                                               parent_adjust=Program.library.parent_adjust)
            # TBD: Return dangling as part of JIT function above
            arities = np.array([Program.library.arities[t] for t in tokens])
            obs_dangling[i, :] = 1 + np.cumsum(arities - 1)

            programs.append(from_tokens(tokens, on_policy=False))

        # Compute priors
        if self.train_n > 0:
            priors = self.prior.at_once(actions, obs_parent, obs_sibling)
        else:
            priors = np.zeros((len(programs), self.max_length, L), dtype=np.float32)

        obs = np.stack([obs_action, obs_parent, obs_sibling, obs_dangling], axis=1)

        return programs, actions, obs, priors

    def __call__(self, actions):
        """
        Parameters
        ----------

        actions : np.ndarray
            Actions to use as starting population.
        """
        t1 = time.perf_counter()

        # TBD: Fix hack
        self.max_length = actions.shape[1]

        # Reset the HOF
        if self.hof is not None:
            self.hof = tools.HallOfFame(maxsize=self.train_n)

        # TODO: Decide whether to select initial population from DSO batch randomly or via
        #       tournament selection. We need to empirically evaluate both before deciding.
        #       We can also leave it to the user to define.

        # if the case, select a subset of samples form GP's initial population
        if self.population_size < actions.shape[0]:
            samples_idx = np.random.choice(actions.shape[0], self.population_size, replace=False)
        else:
            samples_idx = np.arange(actions.shape[0])

        # Get DSO generated batch members into Deap based "individuals"
        # TBD: Can base class of Individual be initialized with tokens and Program?
        individuals = list()
        for idx in samples_idx:
            individuals.append(self.creator.Individual(actions[idx], self.pset,
                                                       self.max_mutations,
                                                       self.ind_representation,
                                                       self.master_sequence))

        self.algorithm.set_population(individuals)

        # Run GP generations
        self.nevals = 0
        # TODO: check for convergence, maybe not all generations are needed
        for i in range(self.generations):
            nevals = self.algorithm(self.hof, i) # Run one generation
            self.nevals += nevals
            
        self.total_nevals += self.nevals

        # Get the HOF batch
        if self.train_n > 0:
            programs, actions, obs, priors = self.get_hof_programs()

        timer = time.perf_counter() - t1
            
        self.verbose_print(timer, programs)

        return programs, actions, obs, priors
    
    def verbose_print(self, timer, programs):
        
        if self.verbose:
            print()
            print("--------------------------------------------------")
            print("GP Time: {:.6}".format(timer))
            print()
            print("Unique expressions evaluated: {}".format(len(Program.cache)))
            print("Total expression evaluations: {}".format(self.total_nevals))
            print("--------------------------------------------------")
            print()
            print("GP -> best program this iteration:")
            print()
            print(programs[0].print_stats())
            print()
            print("--------------------------------------------------")
            print()


    def __del__(self):
        del self.creator.FitnessMin
