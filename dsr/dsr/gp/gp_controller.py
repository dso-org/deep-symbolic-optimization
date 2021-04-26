import multiprocessing

import numpy as np
from pathos.multiprocessing import ProcessPool
from deap import gp
from deap import base
from deap import tools
from deap import creator

from dsr.subroutines import jit_parents_siblings_at_once
from dsr.gp import base as gp_base
from dsr.program import Program, from_tokens
import dsr.gp.utils as U


class GPController:

    def __init__(self, prior, config_gp_meld):
        '''
        It would be nice if json supported comments, then we could put all this there. 
            
        # Constant for now, add to config later
        p_crossover             = 0.25      # Default 0.5: P of crossing two members
        p_mutate                = 0.5       # Default 0.1: P of mutating a member
        seed                    = 0         # Random number seed.
        verbose                 = True      # Print out stats and things.
        max_len                 = 30        # Max expression length for gp. Ideally should match RL max length
        min_len                 = 4         # Min expression length for gp. Ideally should match RL max length
        steps                   = 20        # How many gp steps to do for each DSR epoch. 5 to 20 seems like a good range. 
        tournament_size         = 5         # Default 3: A larger number can converge faster, but me be more biased?
        train_n                 = 10        # How many GP observations to return with RL observations. These still get trimmed if scores are poor later on. 0 turns off return. 
        mutate_tree_max         = 2         # Default 2: How deep can an inserted mutation try be? Deeper swings more wildly. 5 is kind of crazy. Turn up with frustration?
        '''
        
        assert isinstance(config_gp_meld, dict)
                                        
        self.prior = prior
        self.pset = U.create_primitive_set(Program.library)

        self.train_n = config_gp_meld["train_n"]
        self.return_gp_obs = self.train_n > 0

        # Create a Hall of Fame object
        if self.train_n > 0:
            self.hof = tools.HallOfFame(maxsize=self.train_n)
        
        # Create a DEAP toolbox, use generator that takes in RL individuals  
        self.toolbox, self.creator  = self._create_toolbox(self.pset,
                                                           parallel_eval       = config_gp_meld["parallel_eval"], 
                                                           max_len             = config_gp_meld["max_len"], 
                                                           min_len             = config_gp_meld["min_len"], 
                                                           tournament_size     = config_gp_meld["tournament_size"], 
                                                           mutate_tree_max     = config_gp_meld["mutate_tree_max"]) 
        
        # Population will be filled with RL individuals
        _pop                        = []
        
        # create stats widget
        self.mstats                 = gp_base.create_stats_widget()
        
        # Actual loop function that runs GP
        self.algorithm = gp_base.RunOneStepAlgorithm(population     = _pop,
                                                                  toolbox        = self.toolbox,
                                                                  cxpb           = config_gp_meld["p_crossover"],
                                                                  mutpb          = config_gp_meld["p_mutate"],
                                                                  stats          = self.mstats,
                                                                  verbose        = config_gp_meld["verbose"]
                                                                  )   
        
        self.config_gp_meld         = config_gp_meld        
        self.nevals                 = 0

    def check_constraint(self, individual):
        actions, parents, siblings = U.individual_to_dsr_aps(individual, Program.library)
        return self.prior.is_violated(actions, parents, siblings)
        
    def _create_toolbox(self, pset,
                             tournament_size=3, max_len=30, min_len=4,
                             mutate_tree_max=5,
                             parallel_eval=True):
        r"""
            This creates a Deap toolbox with options we set.
        """
        
        assert isinstance(pset, gp.PrimitiveSet),   "pset should be a PrimitiveSet"
        
        # NOTE from deap.creator.create: create(name, base, **kargs):
        # ALSO: Creates a new class named *name* inheriting from *base* 
        
        # Create custom fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Adds fitness into PrimitiveTree
    
        # NOTE from deap.base.Toolbox:  def register(self, alias, function, *args, **kargs):
        # ALSO the function in toolbox is defined as: partial(function, *args, **kargs)
    
        # Define the evolutionary operators
        toolbox = base.Toolbox()
        toolbox.register("select",      tools.selTournament, tournsize=tournament_size)
        toolbox.register("mate",        gp.cxOnePoint)
        toolbox.register("expr_mut",    gp.genFull, min_=0, max_=mutate_tree_max)
        toolbox.register('mutate',      U.multi_mutate, expr=toolbox.expr_mut, pset=pset)
    
        toolbox.decorate("mate", U.staticLimit(key=self.check_constraint, max_value=0))
        toolbox.decorate("mutate", U.staticLimit(key=self.check_constraint, max_value=0))
        
        #overide the built in map function in toolbox
        if parallel_eval:
            print("GP Controller using parallel evaluation via Pathos")
            pool = ProcessPool(nodes = multiprocessing.cpu_count())
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
            # TBD: Off by one in time index? Need initial priors somewhere...
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

        # Get DSR generated batch members into Deap based "individuals"
        # TBD: Can base class of Individual can be initialized with tokens and Program?
        individuals = [self.creator.Individual(U.tokens_to_DEAP(a, self.pset)) for a in actions]
        self.algorithm.set_population(individuals)

        if self.config_gp_meld["verbose"]:
            print(self.algorithm.str_logbook(header_only=True))

        # Run GP generations
        self.nevals = 0
        for i in range(self.config_gp_meld["steps"]):
            nevals = self.algorithm(self.hof) # Run one generation
            self.nevals += nevals

        # Get the HOF batch
        if self.train_n > 0:
            programs, actions, obs, priors = self.get_hof_programs()

        return programs, actions, obs, priors

    def __del__(self):
        del self.creator.FitnessMin
