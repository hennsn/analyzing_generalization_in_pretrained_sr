import sklearn
import numpy as np
import sympy
import numbers

from sklearn.linear_model import LinearRegression

class LinReg(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on Linear Regression
    '''
    def __init__(self, regr = None, X = None, y = None, verbose:int = 0, random_state:int = 0, **kwargs):
        
        if regr is None:
            self.regr = LinearRegression()
        else:
            self.regr = regr
        self.X = X
        self.y = y
        self.verbose = verbose
        self.random_state = random_state
        
    def fit(self, X, y, verbose=0):
        y = y.flatten()
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        self.regr.fit(self.X, self.y)
        return self

    def predict(self, X):
        assert self.X is not None, 'call .fit() first!'
        pred = self.regr.predict(X)
        return pred
    
    def model(self):
        assert self.X is not None
        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]
        expr = self.regr.intercept_
        for x_name, alpha in zip(names, self.regr.coef_):
            expr += alpha*x_name
        return expr
