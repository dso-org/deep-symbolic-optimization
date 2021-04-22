import random
from functools import wraps
from itertools import chain
from collections import defaultdict
import numpy as np
import time
import copy
import operator

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

from deap.gp import PrimitiveSet

from dsr.program import from_tokens, Program, _finish_tokens
from dsr.subroutines import jit_parents_siblings_at_once


def staticLimit(key, max_value):
    """A fixed version of deap.gp.staticLimit that samples without replacement.
    This prevents returning identical objects, for example if both children of a
    crossover operation are illegal."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:

                    # Make another deep copy of the sampled item. This ensures
                    # we don't sample the same object twice.
                    new_inds[i] = copy.deepcopy(random.choice(keep_inds))

            return new_inds
        return wrapper
    return decorator


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
        This needs to be called in a derived task such as gp_regression.
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
        
    def _eval(self, population, halloffame, toolbox):
        
        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        invalid_ind     = [ind for ind in population if not ind.fitness.valid]

        for ind in invalid_ind:
            actions = [t.name for t in ind]
            actions = np.array(actions, dtype=np.int32)
            p = from_tokens(actions, optimize=True, n_objects=1, on_policy=False) # TBD: Support multi-objects
            ind.fitness.values = (-p.r,)
    
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)
            
        return population, halloffame, invalid_ind
            
    def _header(self, population, toolbox, stats=None,
                halloffame=None, verbose=__debug__):
        
        logbook                             = tools.Logbook()
        logbook.header                      = ['gen', 'nevals', 'timer'] + (stats.fields if stats and population else [])
    
        population, halloffame, invalid_ind = self._eval(population, halloffame, toolbox)
    
        record                              = stats.compile(population) if stats and population else {}
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
            record                              = stats.compile(population) if stats and population else {}
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
        record                                      = self.stats.compile(self.population) if self.stats and self.population else {}
        
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


r"""
    Fast special case version of below. This is mainly used during constraint 
    checking. 
"""
def DEAP_to_tokens(individual):

    tokens = np.array([i.name for i in individual], dtype=np.int32)
    
    return tokens

r"""
    This is a base class for accessing DEAP and interfacing it with DSR. 
        
    These are pure symblic components which relate to any symblic task. These are not purely task agnostic and
    are kept seprate from core.
"""
def DEAP_to_padded_tokens(individual, tokens_size):

    if len(individual) > tokens_size:
        print(individual)
        print([t.name for t in individual])

    # Compute unpadded actions
    actions = DEAP_to_tokens(individual)

    actions_padded = np.zeros(tokens_size, dtype=np.int32)
    actions_padded[:len(actions)] = actions

    return actions_padded


def tokens_to_DEAP(tokens, pset):
    """
    Transforms DSR standard tokens into DEAP format tokens.

    DSR and DEAP format are very similar, but we need to translate it over. 

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    pset : gp.PrimitiveSet

    Returns
    _______
    individual : gp.PrimitiveTree
        This is a specialized list that contains points to element from pset that were mapped based 
        on the translation of the tokens. 
    """
        
    assert gp is not None, "Must import Deap GP library to use method. You may need to install it."
    assert isinstance(tokens, np.ndarray), "Raw tokens are supplied as a numpy array."
    assert isinstance(pset, PrimitiveSet), "You need to supply a valid primitive set for translation."
    assert Program.library is not None, "You have to have an initial program class to supply library token conversions."
    
    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    tokens      = _finish_tokens(tokens)

    plist = [pset.mapping[t] for t in tokens]

    individual = gp.PrimitiveTree(plist)
    
    return individual


def individual_to_dsr_aps(individual, library):
    r"""
        This will convert a deap individual to a DSR action, parent, sibling group.
    """ 

    # Get the action tokens from individuals 
    actions = np.array([t.name for t in individual], dtype=np.int32)

    # Add one dim at the front to be (1 x L)
    actions = np.expand_dims(actions, axis=0) 

    # Get the parent/siblings for 
    parent, sibling     = jit_parents_siblings_at_once(actions, arities=library.arities, parent_adjust=library.parent_adjust)
    
    return actions, parent, sibling
        

def create_primitive_set():
    """Create a DEAP primitive set
    """

    assert Program.library is not None, "Library must be set first."

    lib = Program.library
    pset = gp.PrimitiveSet("MAIN", len(lib.input_tokens))
    rename_kwargs = {"ARG{}".format(i) : i for i in range(len(lib.input_tokens))}
    for k, v in rename_kwargs.items():
        pset.mapping[k].name = v # pset.renameArguments doesn't actually rename the Primitive. It just affects the pset mapping
    pset.renameArguments(**rename_kwargs)        

    for i, token in enumerate(lib.tokens):
        if token.input_var is None:
            pset.addPrimitive(None, token.arity, name=i)

    return pset


# This is called when we mate or mutate and individual
def check_constraint(max_length, min_length, max_depth, joint_prior_violation):
    """Check a varety of constraints on a memeber. These include:
        Max Length, Min Length, Max Depth, Trig Ancestors and inversion repetse. 
        
        This is a decorator function that attaches to mutate or mate functions in
        DEAP.
                
        >>> This has been tested and gives the same answer as the old constraint 
            function given trig and inverse constraints. 
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds   = [copy.deepcopy(ind) for ind in args]      # The input individual(s) before the wrapped function is called 
            new_inds    = list(func(*args, **kwargs))               # Calls the wrapped function and returns results
                        
            for i, ind in enumerate(new_inds):
                
                l = len(ind)
                
                if l > max_length:
                    new_inds[i] = random.choice(keep_inds)
                elif l < min_length:
                    new_inds[i] = random.choice(keep_inds)
                elif operator.attrgetter("height")(ind) > max_depth:
                    new_inds[i] = random.choice(keep_inds)
                else:  
                    if joint_prior_violation(new_inds[i]):
                        new_inds[i] = random.choice(keep_inds)                    
            return new_inds

        return wrapper

    return decorator


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

    return hof[0], logbook
