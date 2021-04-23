import multiprocessing
from operator import attrgetter

import numpy as np
from pathos.multiprocessing import ProcessPool
from deap import gp
from deap import base
from deap import tools
from deap import creator

from dsr.subroutines import jit_parents_siblings_at_once
from dsr.gp import base as gp_base
from dsr.prior import make_prior
from dsr.program import Program, from_tokens
from dsr.gp.base import DEAP_to_padded_tokens, tokens_to_DEAP, create_primitive_set, individual_to_dsr_aps

class GPController:
    
    def __init__(self, config_gp_meld, config_prior):
        
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
        assert isinstance(config_prior, dict)
                                        
        # Put the DSR tokens into DEAP format
        self.pset = create_primitive_set()

        # Create a Hall of Fame object
        self.hof = tools.HallOfFame(maxsize=1)
        
        # Create widget for checking constraint violations inside Deap. 
        self.prior = make_prior(Program.library, config_prior)

        self.train_n = config_gp_meld["train_n"]
        
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
        self.algorithms             = gp_base.RunOneStepAlgorithm(population     = _pop,
                                                                  toolbox        = self.toolbox,
                                                                  cxpb           = config_gp_meld["p_crossover"],
                                                                  mutpb          = config_gp_meld["p_mutate"],
                                                                  stats          = self.mstats,
                                                                  halloffame     = self.hof,
                                                                  verbose        = config_gp_meld["verbose"]
                                                                  )   
        
        self.config_gp_meld         = config_gp_meld        
        self.halloffame             = []
        self.population             = []
        self.logbook                = []
        self.nevals                 = 0
        self.return_gp_obs          = None
        
        self.get_top_program        = None
                    
        self.train_n                = self.config_gp_meld["train_n"]

    def check_constraint(self, individual):
        actions, parents, siblings = individual_to_dsr_aps(individual, Program.library)
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
        toolbox.register('mutate',      gp_base.multi_mutate, expr=toolbox.expr_mut, pset=pset)
    
        toolbox.decorate("mate", gp_base.staticLimit(key=self.check_constraint, max_value=0))
        toolbox.decorate("mutate", gp_base.staticLimit(key=self.check_constraint, max_value=0))
        
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
    
    def get_top_n_programs(self, population, actions, prior):
        
        """ Get the top n members of the population, We will also do some things like remove 
            redundant members of the population, which there tend to be a lot of.
            
            Next we compute DSR compatible parents, siblings and actions.  
        """  

        
        # Highest to lowest sorting.
        population      = sorted(population, key=attrgetter('fitness'), reverse=True)
        
        p_items         = []
        p_items_val     = []
        tot             = 0
        
        # Get rid of duplicate members. Make sure these are all unique. 
        for i,p in enumerate(population):
            # we have to check because population members are not nessesarily unique
            if str(p) not in p_items_val:
                p_items.append(p)
                p_items_val.append(str(p))
                tot += 1
                
            if tot == self.train_n:
                break
            
        population          = p_items    
    
        max_tok             = Program.library.L
        
        deap_parent         = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
        deap_sibling        = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
        deap_action         = np.empty((len(population),actions.shape[1]), dtype=np.int32)
        deap_obs_action     = np.empty((len(population),actions.shape[1]), dtype=np.int32)
        
        deap_action[:,0]    = max_tok
        deap_program        = []
        
        # Take each members and get the nesseasry DSR components such as siblings and observations. 
        for i,p in enumerate(population):
            tokens = DEAP_to_padded_tokens(p, actions.shape[1])
            deap_obs_action[i,1:] = tokens[:-1]
            deap_action[i,:] = tokens
            
            deap_program.append(from_tokens(tokens, optimize=False, on_policy=False))
            
            deap_parent[i,:], deap_sibling[i,:] = jit_parents_siblings_at_once(np.expand_dims(tokens, axis=0), 
                                                                               arities=Program.library.arities, 
                                                                               parent_adjust=Program.library.parent_adjust)
                               
        deap_obs_action[:,0]    = max_tok
        deap_obs                = [deap_obs_action, deap_parent, deap_sibling]
        
        # We can generate the priors needed for the update/training step of the DSR network. 
        if self.train_n > 0:
            dp                      = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
            ds                      = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
            dp[:,:-1]               = deap_parent[:,1:]
            ds[:,:-1]               = deap_sibling[:,1:]
            deap_priors             = prior.at_once(deap_action, dp, ds)
        else:
            deap_priors             = np.zeros((len(deap_program), deap_action.shape[1], max_tok), dtype=np.float32)
                
        return deap_program, deap_obs, deap_action, deap_priors
            
    def __call__(self, actions):
        
        assert callable(self.get_top_n_programs)
        assert isinstance(actions, np.ndarray)
        
        # Get DSR generated batch members into Deap based "individuals" 
        # TBD: Can base class of Individual can be initialized with tokens and Program?
        individuals = [self.creator.Individual(tokens_to_DEAP(a, self.pset)) for a in actions]
        self.algorithms.set_population(individuals)
        
        if self.config_gp_meld["verbose"]:
            print(self.algorithms.str_logbook(header_only=True)) # print header
            
        self.halloffame  = []
        self.population  = []
        self.logbook     = []
        self.nevals      = 0
        
        # RUN EACH GP STEP
        for i in range(self.config_gp_meld["steps"]):    
            p, l, h, n  = self.algorithms(init_halloffame=True) # Should probably store each HOF
            self.population  = self.population + p
            self.logbook.append(l)
            self.halloffame.append(h)
            self.nevals += n
         
        # Get back the best n members. 
        if self.config_gp_meld["train_n"] > 0:
            deap_programs, deap_obs, deap_actions, deap_priors      = self.get_top_n_programs(self.population, actions, self.prior)
            self.return_gp_obs                                      = True
        else:
            self.return_gp_obs                                      = False
            
        return deap_programs, deap_obs, deap_actions, deap_priors
    
    def __del__(self):
        
        del self.creator.FitnessMin
