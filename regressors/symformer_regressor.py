import os
import sys
import warnings
import numbers
import numpy as np
import torch
import sklearn
import sympy

from sklearn.base import BaseEstimator, RegressorMixin
CODE_DIR = os.path.dirname(os.path.realpath(__file__))

class TransformerVastl(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        self.src_path = os.path.join(CODE_DIR, 'symformer')
        self.verbose = verbose
        self.random_state = random_state

        # https://github.com/vastlik/symformer
        sys.path.insert(0, self.src_path)
        from symformer.model.runner import Runner
        del sys.path[0]
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        
        params = {
            'beamsize' : 128,
            'delete_files' : False,
            'use_pool' : False,
            'optimization_type' : 'no_opt'
        }
        self.runner_univ = Runner.from_checkpoint('symformer-univariate', **params)
        self.runner_biv = Runner.from_checkpoint('symformer-bivariate', **params)
  
        self.X = None
        self.y = None
        self.positives = []
   
    def fit(self, X, y, verbose = 0):
        import tensorflow as tf

        y = y.flatten()

        # X can only be 2 dimensional at most! So we take first 2 dimensions
        assert len(y.shape) == 1
        
        self.y = y.copy()
        self.positives = np.all(X > 0, axis = 0)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if X.shape[1] == 1:
                runner = self.runner_univ
            else:
                runner = self.runner_biv
            
            points = np.column_stack([X[:, :2], y])
            points = tf.convert_to_tensor([points])
            runner.search.convertor.verbose = verbose
            prediction = runner.search.batch_decode(points)

            self.expr = prediction[-1][0]

            # substitute x -> x_0, y -> x_1
            subs_dict = {
                sympy.Symbol('x') : sympy.Symbol('x_0', real = True),
                sympy.Symbol('y') : sympy.Symbol('x_1', real = True)
            }
            self.expr = self.expr.subs(subs_dict)

        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True, real = True))

        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)

        try:
            pred = self.predict(X)
        except NameError:
            self.expr = sympy.sympify('0')
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


import os, sys, threading

# module-level cache + lock
_runner_cache = {}
_runner_lock = threading.Lock()

def _load_runner_once(src_path, checkpoint_name, params):
    # key must be hashable
    key = (os.path.abspath(src_path), checkpoint_name, tuple(sorted(params.items())))
    with _runner_lock:
        if key in _runner_cache:
            return _runner_cache[key]

        # temporarily add src_path for import
        sys.path.insert(0, src_path)
        try:
            from symformer.model.runner import Runner
        finally:
            # remove the inserted path in all cases
            del sys.path[0]

        runner = Runner.from_checkpoint(checkpoint_name, **params)
        _runner_cache[key] = runner
        return runner


class TransformerVastlCache(BaseEstimator, RegressorMixin):
    def __init__(self, verbose: int = 0, random_state: int = 0, **params):
        self.src_path = os.path.join(CODE_DIR, 'symformer')
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        runner_params = {
            'beamsize': 128,
            'delete_files': False,
            'use_pool': False,
            'optimization_type': 'no_opt'
        }

        # load shared runners (fast on subsequent inits)
        self.runner_univ = _load_runner_once(self.src_path, 'symformer-univariate', runner_params)
        self.runner_biv = _load_runner_once(self.src_path, 'symformer-bivariate', runner_params)

        self.X = None
        self.y = None
        self.positives = []
   
    def fit(self, X, y, verbose = 0):
        import tensorflow as tf

        y = y.flatten()

        # X can only be 2 dimensional at most! So we take first 2 dimensions
        assert len(y.shape) == 1
        
        self.y = y.copy()
        self.positives = np.all(X > 0, axis = 0)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if X.shape[1] == 1:
                runner = self.runner_univ
            else:
                runner = self.runner_biv
            
            points = np.column_stack([X[:, :2], y])
            points = tf.convert_to_tensor([points])
            runner.search.convertor.verbose = verbose
            prediction = runner.search.batch_decode(points)

            self.expr = prediction[-1][0]

            # substitute x -> x_0, y -> x_1
            subs_dict = {
                sympy.Symbol('x') : sympy.Symbol('x_0', real = True),
                sympy.Symbol('y') : sympy.Symbol('x_1', real = True)
            }
            self.expr = self.expr.subs(subs_dict)

        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True, real = True))

        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)

        try:
            pred = self.predict(X)
        except NameError:
            self.expr = sympy.sympify('0')
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