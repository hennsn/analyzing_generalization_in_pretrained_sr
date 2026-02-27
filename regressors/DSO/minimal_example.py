import numpy as np
import os
from dso import DeepSymbolicRegressor

def run_dso(X, y, random_state, verbose = 0):
    params = {
        "experiment" : {
            "seed" : random_state
        },
        "task": {
            "task_type" : "regression",
            "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"],
            "metric" : "neg_mse",
            "poly_optimizer_params" : {
                "degree": 2,
                "regressor": "dso_least_squares",
                "regressor_params": {"n_max_terms" : 2},
            }

        },
        "training": {
            "n_samples": 50000,
            "batch_size": 1000,
            "verbose" : verbose,
            "n_cores_batch" : 1,
        },
    }
    regr = DeepSymbolicRegressor(params)
    regr.fit(X, y)
    return regr.program_.sympy_expr, regr.predict(X)

if __name__ == '__main__':
    func = lambda X: np.exp(X[:, 0]) - X[:, 0]**3

    X1 = np.delete(np.arange(-5, 6), 5).reshape(-1, 1) # without 0
    y1 = func(X1)

    X2 = np.arange(-5, 6).reshape(-1, 1) # with 0
    y2 = func(X2)

    for rand_state in range(10):
        print(f"Random state: {rand_state}")
        model1, pred1 = run_dso(X1, y1, rand_state)
        print(f'Model1: {model1}')
        print(f'MSE: {np.mean((pred1 - y1)**2)}')

        model2, pred2 = run_dso(X2, y2, rand_state)
        print(f'Model2: {model2}')
        print(f'MSE: {np.mean((pred2 - y2)**2)}')
