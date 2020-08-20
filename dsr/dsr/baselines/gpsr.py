import random
import operator
import importlib
from functools import partial

import numpy as np

from dsr.functions import function_map
from dsr.const import make_const_optimizer

from . import constraints


GP_MOD = "deap"
OBJECTS = ["base", "gp", "creator", "tools", "algorithms"]
gp = importlib.import_module(GP_MOD + ".gp")
base = importlib.import_module(GP_MOD + ".base")
creator = importlib.import_module(GP_MOD + ".creator")
tools = importlib.import_module(GP_MOD + ".tools")
algorithms = importlib.import_module(GP_MOD + ".algorithms")


class GP():
    """Genetic-programming based symbolic regression class"""

    def __init__(self, dataset, metric="nmse", population_size=1000,
                 generations=1000, n_samples=None, tournament_size=3,
                 p_crossover=0.5, p_mutate=0.1,
                 const_range=[-1, 1], const_optimizer="scipy",
                 const_params=None, seed=0, early_stopping=False,
                 threshold=1e-12, verbose=True, protected=True,
                 # Constraint hyperparameters
                 constrain_const=True,
                 constrain_trig=True,
                 constrain_inv=True,
                 constrain_min_len=True,
                 constrain_max_len=True,
                 constrain_num_const=True,
                 min_length=4,
                 max_length=30,
                 max_const=3
):

        self.dataset = dataset
        self.fitted = False

        assert n_samples is None or generations is None, "At least one of 'n_samples' or 'generations' must be None."
        if generations is None:
            generations = int(n_samples / population_size)

        # Set hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_mutate = p_mutate
        self.p_crossover = p_crossover
        self.seed = seed
        self.early_stopping = early_stopping
        self.threshold = threshold
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
        nmse_test = partial(self.make_fitness("nmse"), y=dataset.y_test, var_y=np.var(dataset.y_test)) # Function of y_hat
        nmse_test_noiseless = partial(self.make_fitness("nmse"), y=dataset.y_test_noiseless, var_y=np.var(dataset.y_test_noiseless)) # Function of y_hat
        self.nmse_test = partial(self.evaluate, optimize=False, fitness=nmse_test, X=dataset.X_test.T) # Function of individual
        self.nmse_test_noiseless = partial(self.evaluate, optimize=False, fitness=nmse_test_noiseless, X=dataset.X_test.T) # Function of individual
        self.success = lambda ind : self.nmse_test_noiseless(ind)[0] < self.threshold # Function of individual

        # Create the primitive set
        pset = gp.PrimitiveSet("MAIN", dataset.X_train.shape[1])

        # Add input variables
        rename_kwargs = {"ARG{}".format(i) : "x{}".format(i + 1) for i in range(dataset.n_input_var)}
        pset.renameArguments(**rename_kwargs)

        # Add primitives
        for op_name in dataset.function_set:
            assert op_name in function_map, "Operation {} not recognized.".format(op_name)

            # Prepend available protected operators with "protected_"
            if protected and not op_name.startswith("protected_"):
                protected_op_name = "protected_{}".format(op_name)
                if protected_op_name in function_map:
                    op_name = protected_op_name

            op = function_map[op_name]
            pset.addPrimitive(op.function, op.arity, name=op.name)

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

        # Define constraints, each defined by a func : gp.Individual -> bool.
        # We decorate mutation/crossover operators with constrain, which
        # replaces a child with a random parent if func(ind) is True.
        constrain = partial(gp.staticLimit, max_value=0) # Constraint decorator
        funcs = []
        if constrain_min_len:
            funcs.append(constraints.make_check_min_len(min_length)) # Minimum length
        if constrain_max_len:
            funcs.append(constraints.make_check_max_len(max_length)) # Maximum length
        if constrain_inv:
            funcs.append(constraints.check_inv) # Subsequence inverse unary operators
        if constrain_trig:
            funcs.append(constraints.check_trig) # Nested trig operators
        if constrain_const and const:
            funcs.append(constraings.check_const) # All children are constants
        if constrain_num_const and const:
            funcs.append(constraints.make_check_num_const(max_const)) # Number of constants
        for func in funcs:
            for variation in ["mate", "mutate"]:
                self.toolbox.decorate(variation, constrain(func))

        # Create the training function
        self.algorithm = algorithms.eaSimple
    

    def evaluate(self, individual, fitness, X, optimize=False):

        if optimize:
            # Retrieve symbolic constants
            const_idxs = [i for i, node in enumerate(individual) if node.name == "const"]

            # HACK: If early stopping threshold has been reached, don't do training optimization
            # Check if best individual has NMSE below threshold on test set
            if self.early_stopping and len(self.hof) > 0 and self.success(self.hof[0]):
                return (999,)

        if optimize and len(const_idxs) > 0:

            # Objective function for evaluating constants
            def obj(consts):                
                for i, const in zip(const_idxs, consts):
                    individual[i] = gp.Terminal(const, False, object)
                    individual[i].name = "const" # For good measure
                f = self.toolbox.compile(expr=individual)
                y_hat = f(*X)
                y = self.dataset.y_train
                return np.mean((y - y_hat)**2)

            # Do the optimization and set the optimized constants
            x0 = np.ones(len(const_idxs))
            optimized_consts = self.const_opt(obj, x0)
            for i, const in zip(const_idxs, optimized_consts):
                individual[i] = gp.Terminal(const, False, object)
                individual[i].name = "const" # This is necessary to ensure the constant is re-optimized in the next generation

        # Execute the individual
        f = self.toolbox.compile(expr=individual)
        with np.errstate(all="ignore"):
            y_hat = f(*X)

        # Check for validity
        if np.isfinite(y_hat).all():
            return (fitness(y_hat=y_hat),)
        else:
            return (np.inf,)


    def train(self):
        """Train the GP"""

        if self.fitted:
            raise RuntimeError("This GP has already been fitted!")

        random.seed(self.seed)

        pop = self.toolbox.population(n=self.population_size)
        self.hof = tools.HallOfFame(maxsize=1)

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
                                      halloffame=self.hof,
                                      verbose=self.verbose)

        self.fitted = True

        # Delete custom classes
        del creator.FitnessMin
        del creator.Individual
        if "const" in dir(gp):
            del gp.const

        return self.hof[0], logbook


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
