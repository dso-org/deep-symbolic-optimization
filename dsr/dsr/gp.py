import random
import operator
import importlib
import copy
from functools import partial

import numpy as np

import sympy

from dsr.functions import function_map
from dsr.const import make_const_optimizer
from dsr.task.regression.dataset import Dataset

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
       
       
        
class GenericAlgorithm:
    
    def __init__(self):
        
        pass
    
    def _eval(self, population, halloffame, toolbox):
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind     = [ind for ind in population if not ind.fitness.valid]
        fitnesses       = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
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



class RunOneStepAlgorithm(GenericAlgorithm):
    
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
        self.logbook.record(gen=self.gen, nevals=len(invalid_ind), **record)
        
        if self.verbose:
            print(self.logbook.stream)
            
        self.gen += 1
    
        return self.population, self.logbook, self.halloffame
    
    def set_population(self, population):
        
        self.population = population
        
        print('Population Size {}'.format(len(self.population)))
        
        #self.logbook, self.halloffame, self.population = self._header(self.population, self.toolbox, self.stats, self.halloffame, self.verbose)
    
    def append_population(self, population, max_size=None):
        
        self.population = population + self.population
        
        if max_size is not None:
            r = len(self.population)-max_size
            for i in range(0,r):
                self.population.pop(random.randrange(len(self.population)))
        
        print('Population Size {}'.format(len(self.population)))
        #self.logbook, self.halloffame, self.population = self._header(self.population, self.toolbox, self.stats, self.halloffame, self.verbose)
    
    
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
        
        f       = self.toolbox.compile(expr=individual)
        y_hat   = f(*X)
        return (fitness(y_hat=y_hat),)
        
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
                f       = self.toolbox.compile(expr=individual)
                y_hat   = f(*self.X_train)
                y       = self.y_train
                return np.mean((y - y_hat)**2)

            # Do the optimization and set the optimized constants
            x0                  = np.ones(len(const_idxs))
            optimized_consts    = self.const_opt(obj, x0)
            
            for i, const in zip(const_idxs, optimized_consts):
                individual[i] = gp.Terminal(const, False, object)
                individual[i].name = "const" # This is necessary to ensure the constant is re-optimized in the next generation

        return self._finish_eval(individual, self.X_train, self.train_fitness)

    def set_toolbox(self,toolbox):
        
        self.toolbox = toolbox    
        
        
        
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
                         const_optimizer="scipy", const_params=None, const=False):
    
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
    if const:
        const_params    = const_params if const_params is not None else {}
        const_opt       = make_const_optimizer(const_optimizer, **const_params)
        pset.addTerminal(1.0, name="const")     
    else:
        const_opt       = None   
        
    return pset, const_opt



def create_toolbox(pset, eval_func, 
                   tournament_size=3, max_depth=17, max_len=None, max_const=None,
                   gen_func=gp.genHalfAndHalf):
    
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
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #toolbox.register("mutate", gp.mutShrink)
    
    if max_depth is not None:
        toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    if max_len is not None:
        toolbox.decorate("mate",   gp.staticLimit(key=len, max_value=max_len))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_len))
    if const and max_const is not None:
        num_const = lambda ind : len([node for node in ind if node.name == "const"])
        toolbox.decorate("mate",   gp.staticLimit(key=num_const, max_value=max_const))
        toolbox.decorate("mutate", gp.staticLimit(key=num_const, max_value=max_const))

    # Create the training function
    return toolbox, creator 



def create_stats_widget():
    
    stats_fit               = tools.Statistics(lambda p : p.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("min", np.min)
    stats_size              = tools.Statistics(len)
    stats_size.register("avg", np.mean)
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
    
    #print(logbook)
    print(hof)

    