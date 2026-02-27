import os
import sys
import warnings
import sympy
import sklearn.base

from dso import DeepSymbolicRegressor
import numpy as np

class uDSR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on deep symbolic regression
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        
        self.verbose = verbose
        self.random_state = random_state
        
        # https://github.com/dso-org/deep-symbolic-optimization
        #self.src_path = os.path.join(CODE_DIR, 'DSO')
        #sys.path.insert(0, self.src_path)
        #from dso import DeepSymbolicRegressor
        #del sys.path[0]

        # Hyperparams for DSO
        function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"]
        params = {
            "experiment" : {
                "seed" : random_state
            },
            "task": {
                "task_type" : "regression",
                "function_set" : function_set,
                "poly_optimizer_params" : {
                    "degree": 3,
                    "coef_tol": 1e-6,
                    "regressor": "dso_least_squares",
                    "regressor_params": {}
                }
            },
            "training": {
                "n_samples": 100000,
                #"batch_size": 1000,
                "verbose" : verbose,
                "n_cores_batch" : 1,
                "early_stopping": True,
            },

        }

        self.regr = DeepSymbolicRegressor() # DeepSymbolicRegressor(params) # 
        
        self.X = None
        self.y = None
        self.positives = []
        
    def fit(self, X, y, verbose=0):
        y = y.flatten()
        assert len(y.shape) == 1
        self.positives = np.all(X > 0, axis = 0)
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regr = self.regr.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None
        pred = self.regr.predict(X)
        return pred.flatten()

    def model(self):
        assert self.X is not None
        expr = self.regr.program_.sympy_expr
        x_symbs = expr.free_symbols
        symb_dict = {}
        for x in x_symbs:
            idx = int(str(x)[1:])
            symb_dict[idx] = x
            
        for i in symb_dict:
            expr = expr.subs(symb_dict[i], sympy.symbols(f'x_{i-1}', real = True, positive = self.positives[i-1]))
        return expr
