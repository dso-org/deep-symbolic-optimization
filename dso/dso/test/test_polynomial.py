import pytest

import numpy as np

from dso.config import load_config
from dso.program import Program, from_str_tokens
from dso.task import make_task
from dso.library import Polynomial
from dso.task.regression.polyfit import regressors, PolyOptimizer, PolyGenerator

np.random.seed(0)

options = {
        "linear_regression": {"fit_intercept": False},
        "lasso": {"alpha": 1e-6, "fit_intercept": False, "max_iter": 200000, "tol": 1e-9},
        "ridge": {"alpha": 1e-6, "fit_intercept": False},
        "dso_least_squares" : {"cutoff_p_value": 0.05, "n_max_terms": 10, "coef_tol": 1e-12},
        "dso_lasso" : {"gamma": 1E-6, "comp_tol": 1E-4, "rtrn_constrnd_ls": True}
    }

coef_tol = 1e-6
rel_tol = 1e-4

n_input_var = 3
degree = 5
n_pts = 500
X_range = 10.0
n_tests = 10

def check_error(poly, X, y, regressor):
    diff = poly(X)
    diff -= y
    rel_err = np.linalg.norm(diff) / np.linalg.norm(y)
    assert rel_err < rel_tol, "\nregressor: {}\npoly(X) = {}".format(regressor, repr(poly))


def test_polyfit():
    X = np.random.uniform(-X_range / 2, X_range / 2, n_input_var * n_pts)
    X = X.reshape((n_pts, n_input_var))
    poly_generator = PolyGenerator(degree, n_input_var)

    for test in range(1, n_tests+1):
        target_poly = poly_generator.generate()
        y = target_poly(X)

        print("\ntest_polyfit #{}: \ny = {}".format(test, target_poly))
        for regressor in regressors:
            poly_optimizer = PolyOptimizer(degree, coef_tol, regressor, options[regressor])
            poly = poly_optimizer.fit(X, y)
            check_error(poly, X, y, regressor)


def test_poly_optimize():
    config = load_config()
    config["task"]["dataset"] = "Poly-4"
    task = make_task(**config["task"])
    Program.set_task(task)
    Program.set_execute(protected=False)

    target_poly = Polynomial([(1, 1, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                              (0, 0, 1, 0, 0, 1, 0, 0, 0, 1)], np.array([12.0, 1.3, -0.05]))
    y = target_poly(task.X_train)

    print("target_poly = {}".format(repr(target_poly)))
    for regressor in regressors:
        task.poly_optimizer = PolyOptimizer(degree, coef_tol, regressor, options[regressor])
        my_program = from_str_tokens(['div', 'sin', 'x4', 'mul', 'sqrt', 'poly', 'exp', 'x7'])
        my_program.r
        my_poly = my_program.traversal[my_program.poly_pos]
        check_error(my_poly, task.X_train, y, regressor)


def test_poly_to_traversal():
    config = load_config()
    config["task"]["dataset"] = "Korns-3"
    task = make_task(**config["task"])
    Program.set_task(task)
    Program.set_execute(protected=False)

    X = np.random.uniform(-X_range / 2, X_range / 2, n_input_var * n_pts)
    X = X.reshape((n_pts, n_input_var))
    poly_generator = PolyGenerator(degree, n_input_var)
    for test in range(n_tests):
        poly = poly_generator.generate()
        equivalent_program = from_str_tokens(poly.to_str_tokens())

        y = equivalent_program.execute(X)
        diff = poly(X)
        diff -= y
        rel_err = np.linalg.norm(diff) / np.linalg.norm(y)
        assert rel_err < 1e-8, \
            "The converted traversal for {} is incorrect!".format(poly)
