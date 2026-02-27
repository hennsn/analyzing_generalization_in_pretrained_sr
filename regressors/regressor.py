import numpy as np
import sklearn.base

class Regressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Sklearn interface.
    '''

    def __init__(self, **kwargs):
        pass

    def fit(self, X:np.ndarray, y:np.ndarray):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
        '''
        pass 

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''
        Predicts values for given samples.

        @Params:
            X... input data (shape n_samples x inp_dim)
        @Returns:
            predictions (shape n_samples)
        '''
        pass 

    def model(self):
        '''
        Symbolic model

        @Returns:
            sympy expression
        '''
        pass 


