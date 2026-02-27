import sklearn
import numpy as np
import sympy
import numbers
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class PolyReg(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Regressor based on Polynomial Regression
    '''
    def __init__(self, degree:int = 2, alpha:float = 0.0, verbose:int = 0, random_state:int = 0, **params):
        self.alpha = alpha
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if self.alpha == 0.0:
            self.regr = LinearRegression()
        else:
            self.regr = Ridge(alpha=self.alpha, max_iter=100000)

        self.X = None
        self.y = None
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
        X_poly = self.poly.fit_transform(self.X)
        self.regr.fit(X_poly, self.y)

    def predict(self, X):
        assert self.X is not None
        X_poly = self.poly.fit_transform(X)
        pred = self.regr.predict(X_poly)
        return pred

    def model(self):
        assert self.X is not None
        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]

        X_idxs = np.arange(self.X.shape[1])
        X_poly = []
        for degree in range(1, self.degree+1):   
            poly_idxs = itertools.combinations_with_replacement(X_idxs, degree)
            for idxs in poly_idxs:
                prod = 1
                for i in idxs:
                    prod = prod*names[i]
                X_poly.append(prod)

        expr = self.regr.intercept_
        for x_name, alpha in zip(X_poly, self.regr.coef_):
            expr += alpha*x_name
        return expr

