import sklearn
import numpy as np
import sympy
import numbers        

# https://github.com/heal-research/pyoperon
import pyoperon
from pyoperon.sklearn import SymbolicRegressor as OperonRegressor

class Operon(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on Operon
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        self.verbose = verbose
        self.random_state = random_state
        self.regressor_operon = OperonRegressor(random_state = random_state)
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

        self.regressor_operon.fit(self.X, self.y)

        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        model_str = self.regressor_operon.get_model_string(self.regressor_operon.model_, names = x_symbs)
        expr = sympy.sympify(model_str)
        for x in expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                expr = expr.subs(x, sympy.Symbol(str(x), positive = True))
        self.expr = expr
        self.exec_func = sympy.lambdify(x_symbs, self.expr)

    def predict(self, X):
        assert hasattr(self, 'expr')

        if not hasattr(self, 'exec_func'):
            x_symbs = [f'x_{i}' for i in range(X.shape[1])]
            self.exec_func = sympy.lambdify(x_symbs, self.expr)

        pred = self.exec_func(*[X[:, i] for i in range(X.shape[1])])
        if isinstance(pred, numbers.Number):
            pred = pred*np.ones(X.shape[0])
        return pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr