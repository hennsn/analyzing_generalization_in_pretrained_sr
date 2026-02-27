import sklearn
import numpy as np
import sympy
import numbers
import warnings

class PySR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on PySR
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, crossover_prob:float = 0.066, **params):
        
        self.verbose = verbose
        self.random_state = random_state
        self.crossover_prob = crossover_prob
        
        # pip install pysr
        # https://github.com/MilesCranmer/PySR
        from pysr import PySRRegressor
        self.tmp_dir = 'pysr_tmp'

        params['progress'] = False
        params['temp_equation_file'] = True
        #params['tempdir'] = self.tmp_dir
        params['delete_tempfiles'] = True

        if 'crossover_probability' not in params:
            params['crossover_probability'] = self.crossover_prob
        if 'verbosity' not in params:
            params['verbosity'] = self.verbose
        if 'random_state' not in params:
            params['random_state'] = random_state
        if 'binary_operators' not in params:
            params['binary_operators'] = ["+", "*", "-", "/"]
        if 'unary_operators' not in params:
            params['unary_operators'] = ["cos", "exp", "sin", "log", "sqrt"]
        self.regr = PySRRegressor(**params)
        self.X = None
        self.y = None
        self.positives = []
        
    def fit(self, X, y):
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
            self.regr = self.regr.fit(X, y)

        self.expr = self.regr.get_best()['equation']
        self.expr = sympy.sympify(self.expr)

        subs_dict = {}
        for s in self.expr.free_symbols:
            str_s = str(s)
            idx = int(str_s[1:])
            subs_dict[s] = sympy.Symbol(f'x_{idx}')
        self.expr = self.expr.subs(subs_dict)

        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))

        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
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
