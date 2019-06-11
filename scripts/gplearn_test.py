import json
from textwrap import indent

from gplearn.genetic import SymbolicRegressor
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from dsr.dataset import Dataset

# # Generate a dataset
# seed = 0
# rng = np.random.RandomState(seed)
# # X_train = rng.uniform(-5,5,1200).reshape(600,2)
# X_train = rng.uniform(0.1,5.9,60).reshape(30,2)

# ##### sin(x^2) + cos(x*y) + sin(y^2) #####
# # y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) + np.sin(X_train[:,1]**2)

# ##### sin(x^2) + cos(x*y)/sin(y^2) #####
# # y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) / np.sin(X_train[:,1]**2)

# ##### sin(x^2) + cos(x*y)*sin(y^2) #####
# # y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) * np.sin(X_train[:,1]**2)

# ##### sin(x^2) + cos(x*y)*sin(y^2) #####
# # y_train = 6 * np.sin(X_train[:,0]) + np.cos(X_train[:,1])

# Load the config file
config_filename = 'config.json'
with open(config_filename, encoding='utf-8') as f:
    config = json.load(f)

config_dataset = config["dataset"]
config_gp = config["gp"]

# Special case: 'init_depth' is a tuple but JSON can't make tuples
if "init_depth" in config_gp:
    config_gp["init_depth"] = tuple(config_gp["init_depth"])

# Create the dataset
dataset = Dataset(**config_dataset)
X, y = dataset.X_train, dataset.y_train

# Define function set
function_set = config_dataset["operators"]

# Create the symbolic regression model
gp = SymbolicRegressor(**config_gp)
# gp = SymbolicRegressor(
#     population_size=1000,
#     generations=1,
#     tournament_size=20,
#     stopping_criteria=0.0,
#     const_range=None, # Changed from default
#     init_depth=(2,6),
#     init_method="half and half",
#     function_set=function_set, # Changed from default
#     metric="mse", # Changed from default
#     parsimony_coefficient=0.001,
#     p_crossover=0.9,
#     p_subtree_mutation=0.01,
#     p_hoist_mutation=0.01,
#     p_point_mutation=0.01,
#     p_point_replace=0.05,
#     max_samples=1.0,
#     feature_names=None,
#     warm_start=False,
#     low_memory=False,
#     n_jobs=1,
#     verbose=1, # Changed from default
#     random_state=0 # Changed from default
#     )

# Fit using genetic algorithm
gp.fit(X, y)

# Retrieve results
print("Reward:", gp._program.raw_fitness_)
print("Program:", str(gp._program))
try: # Parsing serialized program does not always work
    best_str = str(gp._program).replace("X", "x").replace("add", "Add").replace("mul", "Mul")
    sympy_expr = parse_expr(best_str)
    print("{}".format(indent(pretty(sympy_expr), 'Expression: ')))
except:
    pass
