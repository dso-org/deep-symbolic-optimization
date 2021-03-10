import random
from functools import wraps
from itertools import chain
from collections import defaultdict
from operator import attrgetter
import numpy as np
import time

from dsr.program import Program, from_tokens
from dsr.subroutines import parents_siblings

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

r"""
    This is the core base class for accessing DEAP and interfacing it with DSR. 
        
    It is mostly reserved for core DEAP items that are unrelated to any task.
"""



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
    """ 
        Randomly select one of four types of mutation with even odds for each.
    """
    v = np.random.randint(0,4)

    if v == 0:
        individual = gp.mutUniform(individual, expr, pset)
    elif v == 1:     
        individual = gp.mutNodeReplacement(individual, pset)
    elif v == 2:    
        individual = gp.mutInsert(individual, pset)
    elif v == 3:
        individual = gp.mutShrink(individual)
        
    return individual


def popConstraint():
    """
        This needs to be called in a derived task such as gp_regression
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            raise NotImplementedError

        return wrapper

    return decorator

class GenericAlgorithm:
    """ Top level class which runs the GP, this replaces classes like eaSimple since we need 
        more control over how it runs.
    """
    def __init__(self):
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
    # Would this benefit from using process pooling?
    def _eval(self, population, halloffame, toolbox):
        
        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        invalid_ind     = [ind for ind in population if not ind.fitness.valid]
        # Toolbox.map is registered as the built in python map function.
        # Note that registered functions are still wrapped. 
        # Here we just iterate over all invalid_ind using the function in toolbox.evaluate
        # SEE: toolbox in DEAP base.py
        fitnesses       = toolbox.cmap(toolbox.evaluate, invalid_ind)

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
        logbook.header                      = ['gen', 'nevals', 'timer'] + (stats.fields if stats else [])
    
        population, halloffame, invalid_ind = self._eval(population, halloffame, toolbox)
    
        record                              = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        if verbose:
            print(logbook.stream)
            
        return logbook, halloffame, population
    
    # Would this benefit from using process pooling?
    def _var_and(self, population, toolbox, cxpb, mutpb):
 
        offspring = [toolbox.clone(ind) for ind in population]
    
        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Apply mutation on the offspring        
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

        # Start index is set at function call, or is 0 if doing the header

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
                    # Put Chapter over the column label line
                    column = chapters_txt[name][i+offsets[name]]
                else:
                    # Put the column label
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
        
        assert gp is not None, "Did not import gp. Is DEAP installed?"
        
        self.logbook, self.halloffame, self.population = self._header(population, toolbox, stats, halloffame, verbose)
        
        self.toolbox        = toolbox
        self.cxpb           = cxpb
        self.mutpb          = mutpb
        self.stats          = stats
        self.verbose        = verbose
        
        self.gen        = 0
        
    def __call__(self, init_halloffame=False):
    
    
        if init_halloffame:
            self.halloffame = tools.HallOfFame(maxsize=1)
            
        t1                                          = time.perf_counter()
    
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
        
        timer                                       = time.perf_counter() - t1
        
        self.logbook.record(gen=self.gen, nevals=nevals, timer=timer, **record)
        
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

    
def _get_top_n_programs(population, n, actions, max_len, min_len, DEAP_to_tokens, priors_func=None):
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
            
        if tot == n:
            break
        
    population          = p_items    

    max_tok             = Program.library.L
    
    deap_parent         = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
    deap_sibling        = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
    deap_action         = np.empty((len(population),actions.shape[1]), dtype=np.int32)
    deap_obs_action     = np.empty((len(population),actions.shape[1]), dtype=np.int32)
    
    deap_action[:,0]    = max_tok
    deap_program        = []
    deap_tokens         = []
    deap_expr_length    = []
    
    for i,p in enumerate(population):
        dt, oc, dexpl                   = DEAP_to_tokens(p, actions.shape[1])
        deap_tokens.append(dt)
        deap_expr_length.append(dexpl)
        deap_obs_action[i,1:]           = dt[:-1]
        deap_action[i,:]                = dt
        
        deap_program.append(from_tokens(dt, optimize=True, on_policy=False, optimized_consts=oc))
                
        for j in range(actions.shape[1]-1):       
            # Parent and sibling given the current action
            # Action should be alligned with parent and sibling.
            # The current action should be passed into function inclusivly of itself [:j+1]. 
            p, s                    = parents_siblings(np.expand_dims(dt[0:j+1],axis=0), arities=Program.library.arities, parent_adjust=Program.library.parent_adjust)
            deap_parent[i,j+1]      = p
            deap_sibling[i,j+1]     = s
    
    deap_parent[:,0]        = max_tok - len(Program.library.terminal_tokens)
    deap_sibling[:,0]       = max_tok
    deap_obs_action[:,0]    = max_tok
    deap_obs                = [deap_obs_action, deap_parent, deap_sibling]
    
    if priors_func is not None:
        dp                      = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
        ds                      = np.zeros((len(population),actions.shape[1]), dtype=np.int32) 
        dp[:,:-1]               = deap_parent[:,1:]
        ds[:,:-1]               = deap_sibling[:,1:]
        deap_priors             = priors_func(deap_action, dp, ds)
    else:
        deap_priors             = np.zeros((len(deap_tokens), deap_action.shape[1], max_tok), dtype=np.float32)
        
    return deap_program, deap_obs, deap_action, deap_tokens, deap_priors, deap_expr_length 


def DEAP_to_tokens(individual, tokens_size):
    """
        This needs to be called in a derived task such as gp_regression
    """
    
    raise NotImplementedError


def tokens_to_DEAP(tokens, primitive_set):
    """
        This needs to be called in a derived task such as gp_regression
    """
    
    raise NotImplementedError
        

def create_primitive_set(*args, **kwargs):
    """
        This needs to be called in a derived task such as gp_regression
    """
    
    raise NotImplementedError


def convert_inverse_prim(*args, **kwargs):
    """
        This needs to be called in a derived task such as gp_regression
    """
    
    raise NotImplementedError


def stringify_for_sympy(*args, **kwargs):
    """
        This needs to be called in a derived task such as gp_regression
    """
    
    raise NotImplementedError


def create_stats_widget():
    
    # ma are numpy masked arrays that ignore things like inf
    
    stats_fit               = tools.Statistics(lambda p : p.fitness.values)
    stats_fit.register("avg", lambda x : np.ma.masked_invalid(x).mean())
    stats_fit.register("min", np.min)
    stats_size              = tools.Statistics(len)
    stats_size.register("avg", lambda x : np.ma.masked_invalid(x).mean())
    mstats                  = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    
    return mstats


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
