import numpy as np
import multiprocessing
from pathos.multiprocessing import ProcessPool

try:
    from deap import gp
    from deap import base
    from deap import tools
    from deap import creator
    from deap import algorithms
except ImportError:
    gp          = None
    base        = None
    tools       = None
    creator     = None
    algorithms  = None

from dsr.gp import base as gp_base
from dsr.gp import tokens as gp_tokens
from dsr.prior import make_prior
from dsr.program import Program
from dsr.utils import join_obs

class GPController:
    
    def __init__(self, config_gp_meld, config_task, config_training, config_prior, 
                 pset, eval_func, check_constraint, hof, gen_func=gp_base.GenWithRLIndividuals()):
        
        '''    
        It would be nice if json supported comments, then we could put all this there. 
            
        # Constant for now, add to config later
        init_population_size    = 1         # Put in some members to start?
        p_crossover             = 0.25      # Default 0.5: P of crossing two members
        p_mutate                = 0.5       # Default 0.1: P of mutating a member
        seed                    = 0         # Random number seed.
        verbose                 = True      # Print out stats and things.
        max_len                 = 30        # Max expression length for gp. Ideally should match RL max length
        min_len                 = 4         # Min expression length for gp. Ideally should match RL max length
        steps                   = 20        # How many gp steps to do for each DSR epoch. 5 to 20 seems like a good range. 
        rand_pop_n              = 50        # Random population to append to RL actions for GP, 100 is a good number. 0 turns this off. (Bug: needs to be at least 1?)
        pop_pad                 = 0         # We can add actions more than once exanding GP population x many times. Maybe try 3. 0 turns this off. 
        fitness_metric          = "nmse"    # nmse or nrmse
        recycle_max_size        = 0         # If not zero, we hold over GP population from prior epochs. 1500 works well if we want to do this. 
        tournament_size         = 5         # Default 3: A larger number can converge faster, but me be more biased?
        max_depth               = 30        # Defualt 17: This is mainly a widget to control memory usage. Python sets a hard limit of 90.
        train_n                 = 10        # How many GP observations to return with RL observations. These still get trimmed if scores are poor later on. 0 turns off return. 
        mutate_tree_max         = 2         # Default 2: How deep can an inserted mutation try be? Deeper swings more wildly. 5 is kind of crazy. Turn up with frustration?
        max_const               = 3
        constrain_const         = True
        '''
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        assert isinstance(config_gp_meld,dict) 
        assert isinstance(config_task,dict) 
        assert isinstance(config_training,dict) 
        assert isinstance(pset, gp_tokens.PrimitiveSetTyped)
        assert callable(eval_func)
        assert callable(check_constraint)
        assert isinstance(hof, tools.HallOfFame)
        assert callable(gen_func)
                                        
        # Put the DSR tokens into DEAP format
        self.pset                   = pset
        #self.pset, self.const_opt   = create_primitive_set(dataset) ##, const_params=const_params, have_const=have_const)
        # Create a Hall of Fame object
        self.hof                    = hof
        # Create the object/function that evaluates the population                                                      
        self.eval_func              = eval_func
        # Use a generator we can access to plug in RL population
        self.gen_func               = gen_func
        
        self.check_constraint       = check_constraint
        
        # Create widget for checking constraint violations
        self.joint_prior_violation  = make_prior(Program.library, config_prior, use_violation=True, use_deap=True)
        
        # Create a DEAP toolbox, use generator that takes in RL individuals  
        self.toolbox, self.creator  = self._create_toolbox(self.pset, self.eval_func, 
                                                           gen_func            = self.gen_func, 
                                                           parallel_eval       = config_gp_meld["parallel_eval"], 
                                                           max_len             = config_gp_meld["max_len"], 
                                                           min_len             = config_gp_meld["min_len"], 
                                                           tournament_size     = config_gp_meld["tournament_size"], 
                                                           max_depth           = config_gp_meld["max_depth"], 
                                                           max_const           = config_gp_meld["max_const"], 
                                                           constrain_const     = config_gp_meld["constrain_const"],
                                                           mutate_tree_max     = config_gp_meld["mutate_tree_max"]) 
        
        # Put the toolbox into the evaluation function  
        self.eval_func.set_toolbox(self.toolbox)    
                                                      
        # create some random pops, the default is to use these if we run out of RL individuals. 
        _pop                        = self.toolbox.population(n=config_gp_meld["init_population_size"])
        
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
        
        self.get_top_n_programs     = None
        self.get_top_program        = None
        self.tokens_to_DEAP         = None
        
        self.record_best            = config_gp_meld["record_best"]        
        if self.record_best:
            self.record_best_size   = config_gp_meld["record_best_size"] 
            self.deap_programs      = []
            self.deap_obs           = []
            self.deap_actions       = None
            self.deap_priors        = None
        
        if config_gp_meld["compute_priors"]:
            self.prior_func             = make_prior(Program.library, config_prior, use_at_once=True)
        else:
            self.prior_func             = None
        
        
    def _create_primitive_set(self, *args, **kwargs):
        """
            This needs to be called in a derived task such as gp_regression
        """
    
        raise NotImplementedError
    
    def _base_create_toolbox(self, pset, eval_func, 
                             tournament_size=3, max_depth=17, max_len=30, min_len=4,
                             gen_func=gp.genHalfAndHalf, mutate_tree_max=5,
                             popConstraint=None, parallel_eval=True):
    
        assert isinstance(pset, gp_tokens.PrimitiveSet),   "pset should be a PrimitiveSet"
        assert callable(eval_func),                 "evaluation function should be callable"
        assert callable(gen_func),                  "gen_func should be callable"
        
        # NOTE from deap.creator.create: create(name, base, **kargs):
        # ALSO: Creates a new class named *name* inheriting from *base* 
        
        # Create custom fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Adds fitness into PrimitiveTree
    
        # NOTE from deap.base.Toolbox:  def register(self, alias, function, *args, **kargs):
        # ALSO the function in toolbox is defined as: partial(function, *args, **kargs)
    
        # Define the evolutionary operators
        toolbox = base.Toolbox()
        toolbox.register("expr", gen_func, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        
        if callable(popConstraint):
            #toolbox.decorate("individual", popConstraint(self.joint_prior_violation))
            toolbox.decorate("individual", popConstraint())
        
        toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile",     gp.compile, pset=pset)
        toolbox.register("evaluate",    eval_func)
        toolbox.register("select",      tools.selTournament, tournsize=tournament_size)
        toolbox.register("mate",        gp.cxOnePoint)
        toolbox.register("expr_mut",    gp.genFull, min_=0, max_=mutate_tree_max)
        toolbox.register('mutate',      gp_base.multi_mutate, expr=toolbox.expr_mut, pset=pset)
    
        toolbox.decorate("mate",        self.check_constraint(max_len, min_len, max_depth, self.joint_prior_violation))
        toolbox.decorate("mutate",      self.check_constraint(max_len, min_len, max_depth, self.joint_prior_violation))
        
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
    
    def _add_primitives(self, pset, function_map, function_set):        
    
        for k, v in function_map.items():
            if k in function_set:
                pset.addPrimitive(v.function, v.arity, name=v.name)   
        
        return pset
    
    # This method will probably go away at some point unless ...
    def _concat_best(self, deap_programs, deap_obs, deap_actions, deap_priors):
        
        if self.deap_actions is not None:
            # add record on at the end
            # We set these aside so we don't accidentally re-mix the top record members
            # back into the record. 
            _programs       = deap_programs + self.deap_programs
            _obs            = join_obs(deap_obs,            self.deap_obs)
            _actions        = np.append(deap_actions,       self.deap_actions, axis=0)                
            _priors         = np.append(deap_priors,        self.deap_priors, axis=0)
            copy_back = True
        else:
            copy_back = False
    
        # only add if we have not already added it. 
        if deap_programs[0] not in self.deap_programs:
            
            if self.deap_actions is None:
                self.deap_programs      = [deap_programs[0]]
                self.deap_obs           = [np.expand_dims(deap_obs[0][0], axis=0), 
                                           np.expand_dims(deap_obs[1][0], axis=0), 
                                           np.expand_dims(deap_obs[2][0], axis=0)]
                self.deap_actions       = np.expand_dims(deap_actions[0,:], axis=0) 
                self.deap_priors        = np.expand_dims(deap_priors[0,:], axis=0) 
                
            elif len(self.deap_programs) >= self.record_best_size:
                print("PRE self {}".format(self.deap_actions.shape))
                # This can either be a simple buffer or do a comparison 
                # You probably do not want the buffer to be too big since it can prevent exploration. 
                self.deap_programs      = self.deap_programs[1:] + [deap_programs[0]]
                self.deap_obs           = join_obs(self.deap_obs,           deap_obs, pop_front=True, o2_idx=0)
                self.deap_actions       = np.append(self.deap_actions[1:,], np.expand_dims(deap_actions[0,:], axis=0), axis=0)
                self.deap_priors        = np.append(self.deap_priors[1:,],  np.expand_dims(deap_priors[0,:], axis=0), axis=0)
                
            else:
                print("PRE self {}".format(self.deap_actions.shape))
                self.deap_programs      = self.deap_programs + [deap_programs[0]] 
                self.deap_obs           = join_obs(self.deap_obs,      deap_obs, o2_idx=0)
                self.deap_actions       = np.append(self.deap_actions, np.expand_dims(deap_actions[0,:], axis=0), axis=0)                
                self.deap_priors        = np.append(self.deap_priors,  np.expand_dims(deap_priors[0,:], axis=0), axis=0)

        if copy_back:                           
            deap_programs           = _programs
            deap_obs                = _obs 
            deap_actions            = _actions 
            deap_priors             = _priors
            
        return deap_programs, deap_obs, deap_actions, deap_priors
    
    def _call_pre_process(self):
        pass
    
    def _call_post_process(self):
        pass
            
    def __call__(self, actions):
        
        assert callable(self.get_top_n_programs)
        assert callable(self.tokens_to_DEAP)
        
        assert isinstance(actions, np.ndarray)
        
        self._call_pre_process()
        
        individuals = [self.creator.Individual(self.tokens_to_DEAP(a, self.pset)) for a in actions]
        
        if self.config_gp_meld["rand_pop_n"] > 0:
            individuals += self.toolbox.population(n=self.config_gp_meld["rand_pop_n"])
                
        # we can recycle some of the old GP population. 
        if self.config_gp_meld["recycle_max_size"] > 0:  
            self.algorithms.append_population(individuals, max_size=self.config_gp_meld["recycle_max_size"])
        else:
            self.algorithms.set_population(individuals)
        
        for i in range(self.config_gp_meld["pop_pad"]):
            self.algorithms.append_population(individuals)
        
        if self.config_gp_meld["verbose"]:
            print(self.algorithms.str_logbook(header_only=True)) # print header
            
        self.halloffame  = []
        self.population  = []
        self.logbook     = []
        self.nevals      = 0
        
        for i in range(self.config_gp_meld["steps"]):    
            p, l, h, n  = self.algorithms(init_halloffame=True) # Should probably store each HOF
            self.population  = self.population + p
            self.logbook.append(l)
            self.halloffame.append(h)
            self.nevals += n
         
        if self.config_gp_meld["train_n"] > 0:
            deap_programs, deap_obs, deap_actions, deap_priors      = self.get_top_n_programs(self.population, actions, self.config_gp_meld, self.prior_func)
            self.return_gp_obs                                      = True
        else:
            self.return_gp_obs                                      = False
        
        # Keep a record of the best program from each step
        if self.record_best:
            deap_programs, deap_obs, deap_actions, deap_priors  = self._concat_best(deap_programs, deap_obs, deap_actions, deap_priors)
            
        self._call_post_process()
            
        return deap_programs, deap_obs, deap_actions, deap_priors
    
    def __del__(self):
        
        del self.creator.FitnessMin
        ###del self.creator.Individual