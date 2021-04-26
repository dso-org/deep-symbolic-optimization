import random
from itertools import chain
from collections import defaultdict
import time

import numpy as np
from deap import tools

from dsr.program import from_tokens


class GenericAlgorithm:
    """ Top level class which runs the GP, this replaces classes like eaSimple since we need 
        more control over how it runs.
    """
    def __init__(self):
        pass
        
    def _eval(self, population, toolbox):
        
        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        invalid_ind     = [ind for ind in population if not ind.fitness.valid]

        for ind in invalid_ind:
            actions = [t.name for t in ind]
            actions = np.array(actions, dtype=np.int32)
            p = from_tokens(actions, optimize=True, n_objects=1, on_policy=False) # TBD: Support multi-objects
            ind.fitness.values = (-p.r,)
            
        return population, invalid_ind
            
    def _header(self, population, toolbox, stats=None,
                verbose=__debug__):
        
        logbook                             = tools.Logbook()
        logbook.header                      = ['gen', 'nevals', 'timer'] + (stats.fields if stats and population else [])
    
        population, invalid_ind = self._eval(population, toolbox)
    
        record                              = stats.compile(population) if stats and population else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        if verbose:
            print(logbook.stream)
            
        return logbook, population
    
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
                 hof=None, verbose=__debug__):
    
        logbook, population = self._header(population, toolbox, stats, hof, verbose)

        # TBD NEEDED HERE?
        # Update hall of fame
        if hof is not None:
            hof.update(population)
    
        # Begin the generational process
        for gen in range(1, ngen + 1):
            
            # Select the next generation individuals
            offspring                           = toolbox.select(population, len(population))
    
            # Vary the pool of individuals
            offspring                           = self._var_and(offspring, toolbox, cxpb, mutpb)
    
            # Evaluate the individuals with an invalid fitness
            offspring, invalid_ind  = self._eval(offspring, toolbox)
               
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
    def __init__(self, population, toolbox, cxpb, mutpb, stats=None, verbose=__debug__):
        
        super(RunOneStepAlgorithm, self).__init__()
        
        self.logbook, self.population = self._header(population, toolbox, stats, verbose)
        
        self.toolbox        = toolbox
        self.cxpb           = cxpb
        self.mutpb          = mutpb
        self.stats          = stats
        self.verbose        = verbose
        
        self.gen        = 0
        
    def __call__(self, hof=None):

        t1                                          = time.perf_counter()
    
        # Select the next generation individuals
        offspring                                   = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring                                   = self._var_and(offspring, self.toolbox, self.cxpb, self.mutpb)

        # Evaluate the individuals with an invalid fitness
        offspring, invalid_ind = self._eval(offspring, self.toolbox)
           
        # Replace the current population by the offspring
        self.population[:]                          = offspring

        # Update hall of fame
        if hof is not None:
            hof.update(self.population)

        # Append the current generation statistics to the logbook
        record                                      = self.stats.compile(self.population) if self.stats and self.population else {}
        
        # number of evaluations
        nevals                                      = len(invalid_ind)
        
        timer                                       = time.perf_counter() - t1
        
        self.logbook.record(gen=self.gen, nevals=nevals, timer=timer, **record)
        
        if self.verbose:
            print(self.logbook.stream)
            
        self.gen += 1
    
        return nevals
    
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
                             hof=hof,
                             verbose=verbose)

    # Delete custom classes
    del creator.FitnessMin
    del creator.Individual

    return hof[0], logbook
