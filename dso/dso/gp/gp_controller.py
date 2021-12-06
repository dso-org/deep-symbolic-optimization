import time

import numpy as np
from deap import gp
from deap import base
from deap import tools
from deap import creator

from dso.subroutines import jit_parents_siblings_at_once
from dso.gp import base as gp_base
from dso.program import Program, from_tokens
import dso.gp.utils as U

class GPController:

    def __init__(self, prior, pool, verbose=False, generations=20, p_crossover=0.5,
                 p_mutate=0.5, tournament_size=5, mutate_tree_max=3, train_n=50):
        """
        Initialize the GP controller. The controller will computes a set of 
        fittest individuals using genetic programming.  
        
        Parameters
        ----------
        prior : dso.prior.JointPrior
            JointPrior used to determine whether constraints were violated.

        pool : multiprocessing.Pool
            Pool used for parallelized reward evaluation.

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
        """

        self.prior = prior
        self.pool = pool
        self.pset = U.create_primitive_set(Program.library)
        self.verbose = verbose
        self.generations = generations
        self.train_n = train_n
        self.return_gp_obs = self.train_n > 0
        self.nevals = 0
        self.total_nevals = 0

        # Create a Hall of Fame object
        if self.train_n > 0:
            self.hof = tools.HallOfFame(maxsize=self.train_n)

        # Create a DEAP toolbox
        self.toolbox, self.creator = self._create_toolbox(self.pset,
                                                          tournament_size=tournament_size,
                                                          mutate_tree_max=mutate_tree_max)

        # Actual loop function that runs GP
        self.algorithm = gp_base.RunOneStepAlgorithm(toolbox=self.toolbox,
                                                     cxpb=p_crossover,
                                                     mutpb=p_mutate,
                                                     verbose=verbose)


    def check_constraint(self, individual):
        """
        For an individual, check the DSO prior constraints and make sure that none
        are violated.  
        
        Parameters
        ----------
        individual : deap.gp.PrimitiveTree
            The individual to be checked.
        
        """
        
        actions, parents, siblings = U.individual_to_dso_aps(individual, Program.library)
        return self.prior.is_violated(actions, parents, siblings)


    def _create_toolbox(self, pset, tournament_size=3, mutate_tree_max=5):
        """
        Create a deap.base.Toolbox.
        
        Parameters
        ----------
        pset : deap.gp.PrimitiveSet
            This is a library set of the different token types. 
            
        tournament_size : int
            How many individuals will participate in each tournament. 
            
        mutate_tree_max : int
            The maximum depth (size) for a mutation sub-tree. 
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
        map_fn = self.pool.map if self.pool is not None else map
        toolbox.register("cmap", map_fn)

        # Create the training function
        return toolbox, creator


    def get_hof_programs(self):
        """
        Get the current set of most fit individuals. Also computes and returns
        DSO actions, parents, siblings, and priors.  
        
        Parameters
        ----------
        None
        
        """
        
        hof = self.hof
        L = Program.library.L

        # Init all as numpy arrays of integers with default values or empty
        actions = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_action = np.empty((len(hof), self.max_length), dtype=np.int32)
        obs_parent = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_sibling = np.zeros((len(hof), self.max_length), dtype=np.int32)
        obs_dangling = np.ones((len(hof), self.max_length), dtype=np.int32)

        obs_action[:, 0] = L 
        programs = []

        # Compute actions, obs, and programs
        for i, ind in enumerate(hof):
            tokens = U.DEAP_to_padded_tokens(ind, self.max_length)

            actions[i, :] = tokens
            obs_action[i, 1:] = tokens[:-1]
            obs_parent[i, :], obs_sibling[i, :] = jit_parents_siblings_at_once(np.expand_dims(tokens, axis=0),
                                                                               arities=Program.library.arities,
                                                                               parent_adjust=Program.library.parent_adjust)
           
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
        Call the controller and have it do self.generations steps of genetic programming.
        Each step will call dso.gp.base which does the actual mutation, crossover and tournament 
        selection. This will return a set of complete DSO programs as will as actions, obs and priors
        used to train the RNN. 
        
        Parameters
        ----------

        actions : np.ndarray
            Actions to use as starting population.
            
        """
        
        t1 = time.perf_counter()

        self.max_length = actions.shape[1]

        # Reset the HOF
        if self.hof is not None:
            self.hof = tools.HallOfFame(maxsize=self.train_n)

        # Get DSO generated batch members into Deap based "individuals"
        individuals = [self.creator.Individual(U.tokens_to_DEAP(a, self.pset)) for a in actions]
        self.algorithm.set_population(individuals)

        # Run GP generations
        self.nevals = 0
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
        """
        Print some results from this run of n steps of the gp controller. 
        
        Parameters
        ----------

        timer : time.perf_counter
            The current time index. 
            
        programs : list
            A list of dso programs. 
            
        """
        
        if self.verbose:
            print()
            print("--------------------------------------------------")
            print("GP Time: {:.6}".format(timer))
            print()
            print("Unique expressions evaluated: {}".format(len(Program.cache)))
            print("Total expression evaluations: {}".format(self.total_nevals))
            print("--------------------------------------------------")
            print()
            print("GP -> best program this epoch:")
            print()
            print(programs[0].print_stats())
            print()
            print("--------------------------------------------------")
            print()
            

    def __del__(self):
        del self.creator.FitnessMin
