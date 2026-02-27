# Regressor Interface
import sys
import os
sys.path.insert(0, 'nesymres/src')

import sklearn
import os
import re

from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
import sympy as sp
import json

# Define a function to replace x_i with x_(i-1)
def reduce_index(formula):
    # Use re.sub to find all occurrences of x_<number> and replace with x_<number-1>
    formula_reduced = re.sub(r'x_(\d+)', lambda match: f"x_{int(match.group(1)) - 1}", formula)
    return formula_reduced

class TransformerBiggio(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, random_state:int = 0):
        self.expr = None
        self.random_state = random_state

        if random_state is not None:
            #np.random.seed(random_state)
            torch.manual_seed(random_state)

        ## Load equation configuration and architecture configuration
        import omegaconf
        with open('./nesymres/jupyter/100M/eq_setting.json', 'r') as json_file:
            eq_setting = json.load(json_file)

        cfg = omegaconf.OmegaConf.load("./nesymres/jupyter/100M/config.yaml")

        ## Set up BFGS load rom the hydra config yaml
        self.bfgs = BFGSParams(
                activated = cfg.inference.bfgs.activated,
                n_restarts = cfg.inference.bfgs.n_restarts,
                add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
                normalization_o=cfg.inference.bfgs.normalization_o,
                idx_remove=cfg.inference.bfgs.idx_remove,
                normalization_type=cfg.inference.bfgs.normalization_type,
                stop_time=cfg.inference.bfgs.stop_time,
            )

        self.params_fit = FitParams(word2id=eq_setting["word2id"],
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()},
                                una_ops=eq_setting["una_ops"],
                                bin_ops=eq_setting["bin_ops"],
                                total_variables=list(eq_setting["total_variables"]),
                                total_coefficients=list(eq_setting["total_coefficients"]),
                                rewrite_functions=list(eq_setting["rewrite_functions"]),
                                bfgs=self.bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
        )
        # IMPORTANT: create weights folder under src -> then upload/download the weights there using gdown
        weights_path = "./nesymres/src/100M.ckpt"

        ## Load architecture, set into eval mode, and pass the config parameters
        # IMPORTANT: do not name a variable model, otherwise we cannot call the interface function model()!! -> here we name it transformer_model
        self.transformer = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
        self.transformer.eval()
        if torch.cuda.is_available():
            self.transformer.cuda()

        self.fitfunc = partial(self.transformer.fitfunc, cfg_params = self.params_fit)

    def fit(self, X, y):
        # IMPORTANT: X can only be maximum 3 dimensional
        # flatten y
        y = y.flatten()

        output = self.fitfunc(X, y)
        
        expr_est_str = output['best_bfgs_preds'][0]

        # IMPORTANT: the method returns indices of each variable x_i starting from x_1 -> we need to reset this to x_0
        expr_est_str = reduce_index(expr_est_str)

        expr_est = sp.sympify(expr_est_str)
        self.expr = expr_est
        self.expr_str = expr_est_str

    def predict(self, X):
        task_num_inputs = X.shape[1]
        total_variables = [f'x_{i}' for i in range(task_num_inputs)]
        X_dict = {x: X[:, idx] for idx, x in enumerate(total_variables)}
        y_pred = sp.lambdify(",".join(total_variables), self.expr)(**X_dict)
        return y_pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr