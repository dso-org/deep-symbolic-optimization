import random
import operator
from functools import partial

import numpy as np
from deap import base, gp, creator, tools, algorithms
from deap.gp import *

from dsr.functions import _function_map
from dsr.const import make_const_optimizer


class GP():
    """GP class using DEAP implementation"""

    def __init__(self, dataset, metric="nmse", population_size=1000,
                 generations=1000, tournament_size=3, p_crossover=0.5,
                 p_mutate=0.1, max_depth=17, max_len=None, const_range=[-1, 1],
                 const_optimizer="scipy", const_params=None, seed=0,
                 verbose=True):

        self.dataset = dataset
        self.fitted = False

        # Set hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_mutate = p_mutate
        self.p_crossover = p_crossover
        self.max_depth = max_depth
        self.seed = seed
        self.verbose = verbose

        # Making train/test fitness functions
        fitness = self.make_fitness(metric) 
        fitness_train = partial(fitness, y=dataset.y_train, var_y=np.var(dataset.y_train)) # Function of y_hat
        fitness_test = partial(fitness, y=dataset.y_test, var_y=np.var(dataset.y_test)) # Function of y_hat
        fitness_train_noiseless = partial(fitness, y=dataset.y_train_noiseless, var_y=np.var(dataset.y_train)) # Function of y_hat
        fitness_test_noiseless = partial(fitness, y=dataset.y_test_noiseless, var_y=np.var(dataset.y_test)) # Function of y_hat
        self.eval_train = partial(self.evaluate, optimize=True, fitness=fitness_train, X=dataset.X_train.T) # Function of individual
        self.eval_test = partial(self.evaluate, optimize=False, fitness=fitness_test, X=dataset.X_test.T) # Function of individual
        self.eval_train_noiseless = partial(self.evaluate, optimize=False, fitness=fitness_train_noiseless, X=dataset.X_train.T) # Function of individual
        self.eval_test_noiseless = partial(self.evaluate, optimize=False, fitness=fitness_test_noiseless, X=dataset.X_test.T) # Function of individual
        nrmse = partial(self.make_fitness("nmse"), y=dataset.y_test, var_y=np.var(dataset.y_test)) # Function of y_hat
        self.nrmse = partial(self.evaluate, fitness=nrmse, X=dataset.X_test.T) # Function of individual

        # Create the primitive set
        pset = gp.PrimitiveSet("MAIN", dataset.X_train.shape[1])

        # Add input variables
        rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(dataset.n_input_var)}
        pset.renameArguments(**rename_kwargs)

        # Add primitives
        for k, v in _function_map.items():
            if k in dataset.function_set:
                pset.addPrimitive(v.function, v.arity, name=v.name)        

        # # Add constant
        # if "const" in dataset.function_set:
        #     pset.addEphemeralConstant("const", lambda : random.uniform(const_range[0], const_range[1]))

        # Add constant
        const = "const" in dataset.function_set
        if const:
            const_params = const_params if const_params is not None else {}
            self.const_opt = make_const_optimizer(const_optimizer, **const_params)
            pset.addTerminal(1.0, name="const")

        # Create custom fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # Define the evolutionary operators
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        self.toolbox.register("evaluate", self.eval_train)
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register('mutate', gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        if max_depth is not None:
            self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
            self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
        if max_len is not None:
            self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_len))
            self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_len))
        
        # Create the training function
        self.algorithm = algorithms.eaSimple
    

    def evaluate(self, individual, fitness, X, optimize=False):

        if optimize:
            # Retrieve symbolic constants
            const_idxs = [i for i, node in enumerate(individual) if node.name == "const"]

        if optimize and len(const_idxs) > 0:

            # Objective function for evaluating constants
            def obj(consts):                
                for i, const in zip(const_idxs, consts):
                    individual[i] = Terminal(const, False, object)
                    individual[i].name = "const" # For good measure
                f = self.toolbox.compile(expr=individual)
                y_hat = f(*X)
                y = self.dataset.y_train
                return np.mean((y - y_hat)**2)

            # Do the optimization and set the optimized constants
            x0 = np.ones(len(const_idxs))
            optimized_consts = self.const_opt(obj, x0)
            for i, const in zip(const_idxs, optimized_consts):
                individual[i] = Terminal(const, False, object)
                individual[i].name = "const" # This is necessary to ensure the constant is re-optimized in the next generation

        f = self.toolbox.compile(expr=individual)
        y_hat = f(*X)
        return (fitness(y_hat=y_hat),)


    def train(self):
        """Train the GP"""

        if self.fitted:
            raise RuntimeError("This GP has already been fitted!")

        random.seed(self.seed)

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(maxsize=1)

        stats_fit = tools.Statistics(lambda p : p.fitness.values)
        stats_fit.register("avg", np.mean)
        stats_fit.register("min", np.min)
        stats_size = tools.Statistics(len)
        stats_size.register("avg", np.mean)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        
        pop, logbook = self.algorithm(population=pop,
                                      toolbox=self.toolbox,
                                      cxpb=self.p_crossover,
                                      mutpb=self.p_mutate,
                                      ngen=self.generations,
                                      stats=mstats,
                                      halloffame=hof,
                                      verbose=self.verbose)

        self.fitted = True

        # Delte custom classes
        del creator.FitnessMin
        del creator.Individual
        if "const" in dir(gp):
            del gp.const

        return hof[0], logbook


    def make_fitness(self, metric):
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
