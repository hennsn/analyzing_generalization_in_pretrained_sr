import sympy
import warnings
import numpy as np
import os
import requests
from collections import OrderedDict
import sklearn
import opt_consts
import torch
import numbers
import sympytorch
import sys

class TransformerKamienny(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):

    def __init__(self, verbose:int = 0, random_state:int = 0, rescale:bool = True, **params):

        # https://github.com/facebookresearch/symbolicregression

        from symbolicregression.model import SymbolicTransformerRegressor
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.pt_model = self.load_transformer_()
        self.pt_model.beam_type = 'sampling' # default 'sampling'
        #self.pt_model.max_generated_output_len = 100 # default 200
        self.pt_model.beam_size = 10 #20 # default 10
        self.pt_model.beam_early_stopping = True # default True
        self.regr = SymbolicTransformerRegressor(model=self.pt_model, rescale=rescale) # rescale?
        self.X = None
        self.y = None
        #self.positives = []
        self.verbose = verbose
        self.random_state = random_state

    def load_transformer_(self):
        model_path = os.path.join('symbolicregression', 'model.pt')
        model = None
        try:
            if not os.path.isfile(model_path): 
                url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
                r = requests.get(url, allow_redirects=True)
                open(model_path, 'wb').write(r.content)

            if os.name == 'nt':
                import pathlib
                temp = pathlib.PosixPath
                pathlib.PosixPath = pathlib.WindowsPath    
            model = torch.load(model_path, map_location=torch.device('cpu'))
            
        except Exception as e:
            print("ERROR: model not loaded! path was: {}".format(model_path))
            print(e)    
        
        return model

    def translate_transformer_(self):
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

        if self.regr.tree is None:
            print('zero')
            self.expr = sympy.sympify('0')
            self.func = lambda X: np.zeros((len(X), 1))
        elif self.regr.tree[0] is None: 
            print('zero')
            self.expr = sympy.sympify('0')
            self.func = lambda X: np.zeros((len(X), 1))
        else:
            model_list = self.regr.tree[0][0]
            if "relabed_predicted_tree" in model_list:
                tree = model_list["relabed_predicted_tree"]
            else:
                tree = model_list["predicted_tree"]
            self.func = self.regr.model.env.simplifier.tree_to_numexpr_fn(tree)
                
            model_str = tree.infix()
            for op,replace_op in replace_ops.items():
                model_str = model_str.replace(op,replace_op)
            self.expr = sympy.parse_expr(model_str)
        
        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            #if self.positives[idx]:
            #    self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))

            self.expr = self.expr.subs(x, sympy.Symbol(str(x)))
        
    def fit(self, X, y, verbose = 0):

        y = y.flatten()
        assert len(y.shape) == 1
        self.y = y.copy()

        #self.positives = np.all(X > 0, axis = 0)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        # turnoff warnings 
        warnings.filterwarnings("ignore")

        self.regr.fit(self.X, self.y)
        self.translate_transformer_()

    def predict(self, X):
        assert hasattr(self, 'func')
        pred = self.func(X)[:, 0]
        return pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr
    