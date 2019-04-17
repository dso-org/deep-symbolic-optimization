from gplearn.genetic import SymbolicRegressor
import numpy as np


# Generate a dataset
seed = 0
rng = np.random.RandomState(seed)
X_train = rng.uniform(-5,5,1200).reshape(600,2)

##### sin(x^2) + cos(x*y) + sin(y^2) #####
# y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) + np.sin(X_train[:,1]**2)

##### sin(x^2) + cos(x*y)/sin(y^2) #####
# y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) / np.sin(X_train[:,1]**2)

##### sin(x^2) + cos(x*y)*sin(y^2) #####
y_train = np.sin(X_train[:,0]**2) + np.cos(X_train[:,1]*X_train[:,0]) * np.sin(X_train[:,1]**2)

# Define functino set
function_set = ('add','mul','sin','cos')

# Create the symbolic regression model
ga = SymbolicRegressor(
    population_size=5000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=0,
    function_set=function_set)

# Fit using genetic algorithm
ga.fit(X_train, y_train)

print(ga._program)