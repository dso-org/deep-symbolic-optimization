import random
import operator
import importlib
import copy
import warnings
from functools import partial, wraps
from itertools import chain, compress
from collections import defaultdict
from operator import attrgetter

from numba import jit, prange, int32, void # not yet used

import numpy as np

import sympy

from dsr.functions import function_map, UNARY_TOKENS, BINARY_TOKENS
from dsr.const import make_const_optimizer
from dsr.task.regression.dataset import Dataset
from dsr.program import Program, from_tokens, tokens_to_DEAP, DEAP_to_tokens
from dsr.controller import parents_siblings

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
    
    
class GenWithRLIndividuals:
    """ Forces the generator to select a user provided member first, such as one
        created by RL. Then, when we run out, create them in the usual manner with DEAP.
    """
    def __init__(self, gp_gen_function=gp.genHalfAndHalf):
        
        assert gp is not None, "Did not import gp. Is it installed?"
        assert callable(gp_gen_function), "Generating function must be callable"
        
        self.individuals        = []
        self.gp_gen_function    = gp_gen_function
        
    def __call__(self, pset, min_, max_, type_=None):
        
        if len(self.individuals) > 0:
            return self.individuals.pop()
        else:
            return self.gp_gen_function(pset, min_, max_, type_)
        
    def insert_front(self, individuals):    
        
        assert isinstance(individuals, list), "Individuals must be a list"
        
        self.individuals = individuals + self.individuals
    
    def insert_back(self, individuals):    
        
        assert isinstance(individuals, list), "Individuals must be a list"
        
        self.individuals = self.individuals + individuals
        
    def clear(self):
        
        self.individuals = []
        
        
       
def multi_mutate(individual, expr, pset):   
    """ Randomly select one of four types of mutation with even odds for each.
    """
    v = np.random.randint(0,4)
    
    #v = 0
    
    if v == 0:
        individual = gp.mutUniform(individual, expr, pset)
    elif v == 1:     
        individual = gp.mutNodeReplacement(individual, pset)
    elif v == 2:    
        individual = gp.mutInsert(individual, pset)
    elif v == 3:
        individual = gp.mutShrink(individual)
        
    return individual
        


TRIG_TOKENS = ["sin", "cos", "tan", "csc", "sec", "cot"]

# Define inverse tokens
INVERSE_TOKENS = {
    "exp" : "log",
    "neg" : "neg",
    "inv" : "inv",
    "sqrt" : "n2"
}

# Add inverse trig functions
INVERSE_TOKENS.update({
    t : "arc" + t for t in TRIG_TOKENS
    })

# Add reverse
INVERSE_TOKENS.update({
    v : k for k, v in INVERSE_TOKENS.items()
    })

def check_const(ind):
    """Returns True if children of a parent are all const tokens."""

    names = [node.name for node in ind]
    for i, name in enumerate(names):
        if name in UNARY_TOKENS and names[i+1] == "const":
            return True
        if name in BINARY_TOKENS and names[i+1] == "const" and names[i+1] == "const":
            return True
    return False


def check_inv(names):
    """Returns True if two sequential tokens are inverse unary operators."""

    for i, name in enumerate(names[:-1]):
        if name in INVERSE_TOKENS and names[i+1] == INVERSE_TOKENS[name]:
            return True
    return False

def check_trig(names):
    """Returns True if a descendant of a trig operator is another trig
    operator."""
        
    trig_descendant = False # True when current node is a descendant of a trig operator

    for name in names:
        if name in TRIG_TOKENS:
            if trig_descendant:
                return True
            trig_descendant = True
            trig_dangling   = 1
        elif trig_descendant:
            if name in BINARY_TOKENS:
                trig_dangling += 1
            elif name not in UNARY_TOKENS:
                trig_dangling -= 1
            if trig_dangling == 0:
                trig_descendant = False
                
    return False


def checkConstraint(max_length, min_length, max_depth):
    """Check a varety of constraints on a memeber. These include:
        Max Length, Min Length, Max Depth, Trig Ancestors and inversion repetes. 
        
        This is a decorator function that attaches to mutate or mate functions in
        DEAP.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds   = [copy.deepcopy(ind) for ind in args]
            new_inds    = list(func(*args, **kwargs))
            
            for i, ind in enumerate(new_inds):
                
                l = len(ind)
                
                if l > max_length:
                    new_inds[i] = random.choice(keep_inds)
                elif l < min_length:
                    new_inds[i] = random.choice(keep_inds)
                elif operator.attrgetter("height")(ind) > max_depth:
                    new_inds[i] = random.choice(keep_inds)
                else:    
                    names = [node.name for node in ind]
                    
                    if check_inv(names):
                        new_inds[i] = random.choice(keep_inds)
                    elif check_trig(names):
                        new_inds[i] = random.choice(keep_inds)
                    
            return new_inds

        return wrapper

    return decorator
        
        
class GenericAlgorithm:
    """ Top level class which runs the GP, this replaces classes like eaSimple since we need 
        more control over how it runs.
    """
    def __init__(self):
        
        pass
    
    def _eval(self, population, halloffame, toolbox):
        
        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        invalid_ind     = [ind for ind in population if not ind.fitness.valid]
        fitnesses       = toolbox.map(toolbox.evaluate, invalid_ind)
        
        # If we get back a nan, set it to inf so we never pick it. 
        # We will deal with inf later as needed
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit if np.isfinite(fit) else (np.inf,)
    
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)
            
        return population, halloffame, invalid_ind
            
    def _header(self, population, toolbox, stats=None,
                halloffame=None, verbose=__debug__):
        
        logbook                             = tools.Logbook()
        logbook.header                      = ['gen', 'nevals'] + (stats.fields if stats else [])
    
        population, halloffame, invalid_ind = self._eval(population, halloffame, toolbox)
    
        record                              = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        if verbose:
            print(logbook.stream)
            
        return logbook, halloffame, population
    
    def _var_and(self, population, toolbox, cxpb, mutpb):
 
        offspring = [toolbox.clone(ind) for ind in population]
    
        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
    
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                
                del offspring[i].fitness.values
    
        return offspring
    
    def __call__(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
    
        logbook, halloffame, population = self._header(population, toolbox, stats, halloffame, verbose)
    
        # Begin the generational process
        for gen in range(1, ngen + 1):
            
            # Select the next generation individuals
            offspring                           = toolbox.select(population, len(population))
    
            # Vary the pool of individuals
            offspring                           = self._var_and(offspring, toolbox, cxpb, mutpb)
    
            # Evaluate the individuals with an invalid fitness
            offspring, halloffame, invalid_ind  = self._eval(offspring, halloffame, toolbox)
               
            # Replace the current population by the offspring
            population[:]                       = offspring
    
            # Append the current generation statistics to the logbook
            record                              = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            if verbose:
                print(logbook.stream)
    
        return population, logbook

    def str_logbook(self, header_only=False, startindex=0):
        """
            Pretty print the log book. 
            
            This bypasses the one in DEAP so we can have more control over it. DEAP is
            not made for running inside meta epochs. It does not understand how we will
            call it over and over and so, it prints the logs in a format that really
            does not work here. 
        """
        
        if header_only:
            startindex  = 0
            endindex    = 1
        else:
            endindex    = -1
        
        columns = self.logbook.header
        if not columns:
            columns = sorted(self.logbook[0].keys()) + sorted(self.logbook.chapters.keys())
        if not self.logbook.columns_len or len(self.logbook.columns_len) != len(columns):
            self.logbook.columns_len = map(len, columns)

        chapters_txt = {}
        offsets = defaultdict(int)
        for name, chapter in self.logbook.chapters.items():
            chapters_txt[name] = chapter.__txt__(startindex)
            if startindex == 0:
                offsets[name] = len(chapters_txt[name]) - len(self.logbook)

        str_matrix = []
        
        for i, line in enumerate(self.logbook[startindex:endindex]):
            str_line = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i+offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:n}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                self.logbook.columns_len[j] = max(self.logbook.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)
                    
        if startindex == 0 and self.logbook.log_header:
            header = []
            nlines = 1
            if len(self.logbook.chapters) > 0:
                nlines += max(map(len, chapters_txt.values())) - len(self.logbook) + 1
            header = [[] for i in range(nlines)]
            
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    length = max(len(line.expandtabs()) for line in chapters_txt[name])
                    blanks = nlines - 2 - offsets[name]
                    for i in range(blanks):
                        header[i].append(" " * length)
                    header[blanks].append(name.center(length))
                    header[blanks+1].append("-" * length)
                    for i in range(offsets[name]):
                        header[blanks+2+i].append(chapters_txt[name][i])
                else:
                    length = max(len(line[j].expandtabs()) for line in str_matrix)
                    for line in header[:-1]:
                        line.append(" " * length)
                    header[-1].append(name)
            
            if header_only:
                str_matrix = header  
            else:
                str_matrix = chain(header, str_matrix)
            
            
        template    = "\t".join("{%i:<%i}" % (i, l) for i, l in enumerate(self.logbook.columns_len))
        text        = [template.format(*line) for line in str_matrix]

        return "\n".join(text)
    
    
class RunOneStepAlgorithm(GenericAlgorithm):
    """ Top level class which runs the GP, this replaces classes like eaSimple since we need 
        more control over how it runs.
    """
    def __init__(self, population, toolbox, cxpb, mutpb, stats=None, halloffame=None, verbose=__debug__):
        
        super(RunOneStepAlgorithm, self).__init__()
        
        self.logbook, self.halloffame, self.population = self._header(population, toolbox, stats, halloffame, verbose)
        
        self.toolbox    = toolbox
        self.cxpb       = cxpb
        self.mutpb      = mutpb
        self.stats      = stats
        self.verbose    = verbose
        
        self.gen        = 0
        
    def __call__(self, init_halloffame=False):
    
        if init_halloffame:
            self.halloffame = tools.HallOfFame(maxsize=1)
    
        # Select the next generation individuals
        offspring                                   = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring                                   = self._var_and(offspring, self.toolbox, self.cxpb, self.mutpb)

        # Evaluate the individuals with an invalid fitness
        offspring, self.halloffame, invalid_ind     = self._eval(offspring, self.halloffame, self.toolbox)
           
        # Replace the current population by the offspring
        self.population[:]                          = offspring

        # Append the current generation statistics to the logbook
        record                                      = self.stats.compile(self.population) if self.stats else {}
        
        # number of evaluations
        nevals                                      = len(invalid_ind)
        
        self.logbook.record(gen=self.gen, nevals=nevals, **record)
        
        if self.verbose:
            print(self.logbook.stream)
            
        self.gen += 1
    
        return self.population, self.logbook, self.halloffame, nevals
    
    def set_population(self, population):
        
        self.population = population
        
        if self.verbose:
            print('Population Size {}'.format(len(self.population)))
    
    def append_population(self, population, max_size=None):
        
        if max_size is not None:
            r = len(self.population)-max_size
            if r > 0:
                for i in range(0,r):
                    self.population.pop(random.randrange(len(self.population)))
                
        self.population += population
        
        if self.verbose:
            print('Population Size {}'.format(len(self.population)))
    
    
class GenericEvaluate:
    
    def __init__(self, const_opt, hof, dataset, fitness_metric="nmse",
                 optimize=True, early_stopping=False, threshold=1e-12):
    
        self.toolbox            = None
        
        self.const_opt          = const_opt
        self.hof                = hof
        self.X_train            = dataset.X_train.T
        self.X_test             = dataset.X_test.T
        self.y_train            = dataset.y_train
        self.optimize           = optimize
        self.early_stopping     = early_stopping
        self.threshold          = threshold        
        
        fitness                 = make_fitness(fitness_metric)
        self.train_fitness      = partial(fitness, y=dataset.y_train, var_y=np.var(dataset.y_train))
        self.test_fitness       = partial(fitness, y=dataset.y_test,  var_y=np.var(dataset.y_test)) # Function of y_hat
        
    def _finish_eval(self, individual, X, fitness):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f       = self.toolbox.compile(expr=individual)
            y_hat   = f(*X)
            fit     = (fitness(y_hat=y_hat),)
        
        return fit
        
    def __call__(self, individual):

        assert self.toolbox is not None, "Must set toolbox first."

        if self.optimize:
            # Retrieve symbolic constants
            const_idxs = [i for i, node in enumerate(individual) if node.name == "const"]

            # HACK: If early stopping threshold has been reached, don't do training optimization
            # Check if best individual has NMSE below threshold on test set
            if self.early_stopping and len(self.hof) > 0 and self._finish_eval(self.hof[0], self.X_test, self.test_fitness)[0] < self.threshold:
                return (1.0,)

        if self.optimize and len(const_idxs) > 0:

            # Objective function for evaluating constants
            def obj(consts):                
                for i, const in zip(const_idxs, consts):
                    individual[i] = gp.Terminal(const, False, object)
                    individual[i].name = "const" # For good measure
                #print(individual)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f       = self.toolbox.compile(expr=individual)
                    
                    # Sometimes this evaluation can fail. If so, return largest error possible.
                    try:
                        y_hat   = f(*self.X_train)
                    except:
                        return np.finfo(np.float).max
                    
                    y       = self.y_train
                    res     = np.mean((y - y_hat)**2)

                # Sometimes this evaluation can fail. If so, return largest error possible.
                if np.isfinite(res):
                    return res
                else:
                    return np.finfo(np.float).max

            # Do the optimization and set the optimized constants
            x0                  = np.ones(len(const_idxs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimized_consts    = self.const_opt(obj, x0)
            
            for i, const in zip(const_idxs, optimized_consts):
                individual[i] = gp.Terminal(const, False, object)
                individual[i].name = "const" # This is necessary to ensure the constant is re-optimized in the next generation

        return self._finish_eval(individual, self.X_train, self.train_fitness)

    def set_toolbox(self,toolbox):
        
        self.toolbox = toolbox    
        

class GPController:
    
    def __init__(self, config_gp_meld, config_task, config_training):
        
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
        rand_pop_n              = 50        # Random population to append to RL actions for GP, 100 is a good number. 0 turns this off.
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
            
        config_dataset              = config_task["dataset"]
        dataset                     = Dataset(**config_dataset)
                    
        const_params                = config_training['const_params']
        have_const                  = "const" in dataset.function_set       
                    
        # Put the DSR tokens into DEAP format
        self.pset, self.const_opt   = create_primitive_set(dataset, const_params=const_params, have_const=have_const)
        # Create a Hall of Fame object
        self.hof                    = tools.HallOfFame(maxsize=1) 
        # Create the object/function that evaluates the population                                                      
        self.eval_func              = GenericEvaluate(self.const_opt, self.hof, dataset, fitness_metric=config_gp_meld["fitness_metric"]) 
        # Use a generator we can access to plug in RL population
        self.gen_func               = GenWithRLIndividuals()
        # Create a DEAP toolbox, use generator that takes in RL individuals      
        self.toolbox, self.creator  = create_toolbox(self.pset, self.eval_func, 
                                                 gen_func            = self.gen_func, max_len=config_gp_meld["max_len"], 
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
        self.mstats                 = create_stats_widget()
        
        # Actual loop function that runs GP
        self.algorithms             = RunOneStepAlgorithm(population     = _pop,
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
    
    def __call__(self, actions):
        
        individuals     = [self.creator.Individual(tokens_to_DEAP(a, self.pset)) for a in actions]
        
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
         
        if self.config_gp_meld["train_n"] > 1:
            deap_programs, deap_obs, deap_actions      = get_top_n_programs(self.population, self.config_gp_meld["train_n"], actions)
            return_gp_obs                              = True
        else:
            deap_programs, deap_obs, deap_actions      = get_top_program(self.halloffame, actions)
            return_gp_obs                              = self.config_gp_meld["train_n"]
            
        return deap_programs, deap_obs, deap_actions, return_gp_obs
    
    def __del__(self):
        
        del self.creator.FitnessMin
        del self.creator.Individual

        
def make_fitness(metric):
        """Generates a fitness function by name"""

        if metric == "mse":
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2)

        elif metric == "rmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2))

        elif metric == "nmse":
            fitness = lambda y, y_hat, var_y : np.mean((y - y_hat)**2 / var_y)

        elif metric == "nrmse":
            fitness = lambda y, y_hat, var_y : np.sqrt(np.mean((y - y_hat)**2 / var_y))

        else:
            raise ValueError("Metric not recognized.")

        return fitness
        
        
        
def create_primitive_set(dataset,  
                         const_optimizer="scipy", const_params=None, have_const=False):
    """Create a DEAP primitive set from DSR functions and consts
    """
    
    assert gp is not None,              "Did not import gp. Is it installed?"
    assert isinstance(dataset, object), "dataset should be a DSR Dataset object" 
    
    pset = gp.PrimitiveSet("MAIN", dataset.X_train.shape[1])

    # Add input variables
    rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(dataset.n_input_var)}
    pset.renameArguments(**rename_kwargs)

    # Add primitives
    for k, v in function_map.items():
        if k in dataset.function_set:
            pset.addPrimitive(v.function, v.arity, name=v.name)    
    
    # Are we optimizing a const?               
    if have_const:
        const_params    = const_params if const_params is not None else {}
        const_opt       = make_const_optimizer(const_optimizer, **const_params)
        pset.addTerminal(1.0, name="const")     
    else:
        const_opt       = None   
        
    return pset, const_opt



def create_toolbox(pset, eval_func, 
                   tournament_size=3, max_depth=17, max_len=30, min_len=4, max_const=None, constrain_const=False,
                   gen_func=gp.genHalfAndHalf, mutate_tree_max=5):
    
    assert gp is not None,                      "Did not import gp. Is it installed?"
    assert isinstance(pset, gp.PrimitiveSet),   "pset should be a gp.PrimitiveSet"
    assert callable(eval_func),                 "evaluation function should be callable"
    assert callable(gen_func),                  "gen_func should be callable"
    
    const   = "const" in pset.context
    
    # Create custom fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Adds fitness into PrimitiveTree

    # Define the evolutionary operators
    toolbox = base.Toolbox()
    toolbox.register("expr", gen_func, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_func)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=mutate_tree_max)
    toolbox.register('mutate', multi_mutate, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate",     checkConstraint(max_len, min_len, max_depth))
    toolbox.decorate("mutate",   checkConstraint(max_len, min_len, max_depth))

    if const and max_const is not None:
        assert isinstance(max_const,int)
        assert max_const >= 0
        num_const = lambda ind : len([node for node in ind if node.name == "const"])
        toolbox.decorate("mate",   gp.staticLimit(key=num_const, max_value=max_const))
        toolbox.decorate("mutate", gp.staticLimit(key=num_const, max_value=max_const))

    if const and constrain_const is True:
        toolbox.decorate("mate",   gp.staticLimit(key=check_const, max_value=0))
        toolbox.decorate("mutate", gp.staticLimit(key=check_const, max_value=0))
        
    # Create the training function
    return toolbox, creator 



def create_stats_widget():
    
    # ma are numpy masked arrays that ignore things like inf
    
    stats_fit               = tools.Statistics(lambda p : p.fitness.values)
    stats_fit.register("avg", lambda x : np.ma.masked_invalid(x).mean())
    stats_fit.register("min", np.min)
    stats_size              = tools.Statistics(len)
    stats_size.register("avg", lambda x : np.ma.masked_invalid(x).mean())
    mstats                  = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    
    return mstats



def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    #prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_),
        'inv': lambda *args_: "Pow(-1)".format(*args_),
        'neg': lambda *args_: "Mul(-1)".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)



def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string



def get_top_program(halloffame, actions):
    """ In addition to returning the best program, this will also compute DSR compatible parents, siblings and actions.
    """
    max_tok                         = len(Program.library)
    deap_tokens, deap_expr_length   = DEAP_to_tokens(halloffame[-1][0], actions.shape[1])
       
    deap_parent     = np.zeros(deap_tokens.shape[0], dtype=np.int32) 
    deap_sibling    = np.zeros(deap_tokens.shape[0], dtype=np.int32) 
    
    deap_action     = np.empty(deap_tokens.shape[0], dtype=np.int32)
    deap_action[1:] = deap_tokens[:-1]
    deap_action[0]  = max_tok
    
    for i in range(deap_expr_length-1):       
        p, s                = parents_siblings(np.expand_dims(deap_tokens[0:i+1],axis=0), arities=Program.arities, parent_adjust=Program.parent_adjust)
        deap_parent[i+1]    = p
        deap_sibling[i+1]   = s
    
    deap_parent[0]  = max_tok - len(Program.terminal_tokens)
    deap_sibling[0] = max_tok
    
    deap_obs        = [np.expand_dims(deap_action,axis=0), np.expand_dims(deap_parent,axis=0), np.expand_dims(deap_sibling,axis=0)]
    deap_action     = np.expand_dims(deap_action,axis=0)    
    deap_program    = [from_tokens(deap_tokens, optimize=True)]

    return deap_program, deap_obs, deap_action
    
    
    
def get_top_n_programs(population, n, actions):
    """ Get the top n members of the population, We will also do some things like remove 
        redundant members of the population, which there tend to be a lot of.
        
        Next we compute DSR compatible parents, siblings and actions.  
    """
    scores          = np.zeros(len(population),dtype=np.float)
    
    # Highest to lowest sorting.
    population      = sorted(population, key=attrgetter('fitness'), reverse=True)
    
    p_items         = []
    p_items_val     = []
    tot             = 0
    
    # Get rid of duplicate members. Make sure these are all unique. 
    for i,p in enumerate(population):
        #print(p)
        # we have to check because population members are not nessesarily unique
        if str(p) not in p_items_val:
            p_items.append(p)
            p_items_val.append(str(p))
            tot += 1
            
        if tot == n:
            break
        
    population          = p_items    

    max_tok             = len(Program.library)
    
    deap_parent         = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
    deap_sibling        = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
    deap_action         = np.empty((len(population),actions.shape[1]), dtype=np.int32)
    deap_action[:,0]    = max_tok
    deap_program        = []
    
    for i,p in enumerate(population):
        deap_tokens, deap_expr_length   = DEAP_to_tokens(p, actions.shape[1])
        deap_action[i,1:]               = deap_tokens[:-1]
        deap_program.append(from_tokens(deap_tokens, optimize=True))
        
        for j in range(deap_expr_length-1):       
            p, s                    = parents_siblings(np.expand_dims(deap_tokens[0:j+1],axis=0), arities=Program.arities, parent_adjust=Program.parent_adjust)
            deap_parent[i,j+1]      = p
            deap_sibling[i,j+1]     = s
    
    deap_parent[:,0]    = max_tok - len(Program.terminal_tokens)
    deap_sibling[:,0]   = max_tok

    deap_obs            = [deap_action, deap_parent, deap_sibling]
    
    return deap_program, deap_obs, deap_action


def generic_train(toolbox, hof, algorithm,
                  population_size=1000, p_crossover=0.5, p_mutate=0.1, generations=1000,
                  seed=0, verbose=True):
    
    """Train the GP"""

    random.seed(seed)

    pop         = toolbox.population(n=population_size)
    
    mstats      = create_stats_widget()
    
    pop, logbook = algorithm(population=pop,
                             toolbox=toolbox,
                             cxpb=p_crossover,
                             mutpb=p_mutate,
                             ngen=generations,
                             stats=mstats,
                             halloffame=hof,
                             verbose=verbose)

    # Delete custom classes
    del creator.FitnessMin
    del creator.Individual
    if "const" in dir(gp):
        del gp.const

    return hof[0], logbook


if __name__ == "__main__":
        
    import json
    
    with open("dsr/config.json", encoding='utf-8') as f:
        config = json.load(f)

    # Required configs
    config_task             = config["task"]            # Task specification parameters
    
    config_dataset          = config_task["dataset"]
    config_dataset["name"]  = 'R1'
    dataset                 = Dataset(**config_dataset)

    pset, const_opt         = create_primitive_set(dataset)
    hof                     = tools.HallOfFame(maxsize=1)                   # Create a Hall of Fame object
    eval_func               = GenericEvaluate(const_opt, hof, dataset)      # Create the object/function that evaluates the population
    toolbox                 = create_toolbox(pset, eval_func, max_len=30)   # Create a DEAP toolbox
    algorithms              = GenericAlgorithm()                            # Actual loop function that runs GP
    
    eval_func.set_toolbox(toolbox)                                          # Put the toolbox into the evaluation function
    
    hof, logbook            = generic_train(toolbox, hof, algorithms)
    
    print(hof)

    