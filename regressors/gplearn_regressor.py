import sklearn
import numpy as np
import sympy
import numbers

# https://gplearn.readthedocs.io/en/stable/index.html
# pip install gplearn
import gplearn
from gplearn.genetic import SymbolicRegressor as GPlearnRegressor

class GPlearn(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on gplearn.
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        self.verbose = verbose
        self.random_state = random_state

        self.converter = {
            'add': lambda x, y : x + y,
            'sub' : lambda x, y: x - y,
            'mul': lambda x, y : x * y,
            'div' : lambda x, y: x / y,
            'neg': lambda x : -x,
            'inv': lambda x : 1/x,
            'sin' : lambda x: sympy.sin(x),
            'cos' : lambda x: sympy.cos(x),
            'log' : lambda x: sympy.log(x),
            'sqrt' : lambda x: sympy.sqrt(x),
        }
        funcs = list(self.converter.keys())
        
        params['function_set'] = funcs
        if 'verbose' not in params:
            params['verbose'] = verbose
        if 'random_state' not in params:
            params['random_state'] = random_state

        #params['parsimony_coefficient'] = 0.1

        self.est_gp = GPlearnRegressor(**params)
        
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
        for i in range(self.X.shape[1]):
            self.converter[f'X{i}'] = sympy.symbols(f'x_{i}', real = True)
        self.est_gp.fit(self.X, self.y)

        for i in range(self.X.shape[1]):
            self.converter[f'X{i}'] = sympy.symbols(f'x_{i}', real = True, positive = self.positives[i])
        self.expr = sympy.sympify(str(self.est_gp._program), locals=self.converter)
        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)
        return self

    
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