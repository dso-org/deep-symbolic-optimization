import random
import time

import numpy as np
from deap import tools

from dsr.program import from_tokens


class RunOneStepAlgorithm:
    """
    Top level class which runs the GP one generation at a time, replacing
    classes like eaSimple since we need more control over how it runs.
    """

    def __init__(self, toolbox, cxpb, mutpb, verbose=__debug__):

        self.toolbox = toolbox
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.verbose = verbose

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals', 'timer']

        self.population = None # Must be explicitly set
        self.gen = 0

    def _eval(self, population):

        # Evaluate the individuals with an invalid fitness
        # This way we do not evaluate individuals that we have already seen.
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        for ind in invalid_ind:
            actions = [t.name for t in ind]
            actions = np.array(actions, dtype=np.int32)
            p = from_tokens(actions, optimize=True, n_objects=1,
                            on_policy=False) # TBD: Support multi-objects
            ind.fitness.values = (-p.r,)

        return population, invalid_ind

    # Would this benefit from using process pooling?
    def _var_and(self, population):

        offspring = [self.toolbox.clone(ind) for ind in population]

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])

                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                offspring[i], = self.toolbox.mutate(offspring[i])

                del offspring[i].fitness.values

        return offspring

    def __call__(self, hof=None):

        t1 = time.perf_counter()

        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring = self._var_and(offspring)

        # Evaluate the individuals with an invalid fitness
        offspring, invalid_ind = self._eval(offspring)

        # Replace the current population by the offspring
        self.population[:] = offspring

        # Update hall of fame
        if hof is not None:
            hof.update(self.population)

        # number of evaluations
        nevals = len(invalid_ind)

        timer = time.perf_counter() - t1

        self.logbook.record(gen=self.gen, nevals=nevals, timer=timer)

        if self.verbose:
            print(self.logbook.stream)

        self.gen += 1

        return nevals

    def set_population(self, population):

        self.population = population

        if self.verbose:
            print('Population Size {}'.format(len(self.population)))
