import time
import json
import os
import numbers

import torch
import numpy as np
import sympy as sp
from functools import partial
from sympy import lambdify
import omegaconf

from TPSR.parsers import get_parser
import TPSR.symbolicregression as symbolicregression
from TPSR.symbolicregression.envs import build_env
from TPSR.symbolicregression.model import build_modules
from TPSR.symbolicregression.trainer import Trainer
from TPSR.symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test 
from TPSR.dyna_gym.agents.uct import UCT
from TPSR.dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from TPSR.rl_env import RLEnv
from TPSR.default_pi import E2EHeuristic, NesymresHeuristic
from TPSR.symbolicregression.metrics import compute_metrics
from TPSR.nesymres.src.nesymres.architectures.model import Model
from TPSR.nesymres.src.nesymres.utils import load_metadata_hdf5
from TPSR.nesymres.src.nesymres.dclasses import FitParams, NNEquation, BFGSParams
from TPSR.reward import compute_reward_e2e, compute_reward_nesymres


class TransformerTPSR():

    def __init__(self, random_state:int = 0):
        """
        """
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    def fit(self, X, y, verbose = 0, method="e2e"):

        # set gpu:1 as default gpu 
        torch.cuda.set_device(1) 
        self.case = 1
        parser = get_parser()
        self.params = parser.parse_args()
        np.random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed(self.params.seed)
        self.params.debug = True
        self.params.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {self.params.device}")
        #print(f"params: {self.params}")

        equation_env = build_env(self.params)
        modules = build_modules(equation_env, self.params)
        if not self.params.cpu:
            assert torch.cuda.is_available()
        symbolicregression.utils.CUDA = not self.params.cpu
        trainer = Trainer(modules, equation_env, self.params)
        
        #Example of Equation-Data:
        # x0 = np.random.uniform(-2,2, 200)
        #y= (x0**2 ) * np.sin(5*x0) + np.exp(-0.5*x0)
        #data = np.concatenate((x0.reshape(-1,1),y.reshape(-1,1)), axis=1)
        #samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
        #samples['x_to_fit'] = [data[:,:1]]
        #samples['y_to_fit'] = [data[:,1].reshape(-1,1)]
        #samples['x_to_pred'] = [data[:,:1]]
        #samples['y_to_pred'] = [data[:,1].reshape(-1,1)]

        data = np.concatenate((X,y.reshape(-1,1)), axis=1) 
        samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
        samples['x_to_fit'] = [data[:,:X.shape[1]]]
        samples['y_to_fit'] = [data[:,X.shape[1]].reshape(-1,1)]
        samples['x_to_pred'] = [data[:,:X.shape[1]]]
        samples['y_to_pred'] = [data[:,X.shape[1]].reshape(-1,1)]
        
        #Main
        if method == "e2e":
          expr = main_e2e(self.case, self.params, equation_env, samples)
        elif method == "nesymres":
          expr = main_nesymres(self.case, self.params, equation_env, trainer, samples, X, y)
        self.expr = expr

    def predict(self, X):
        pred = eval_expr(self.expr, X)
        return pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr

def main_e2e(case, params, equation_env, samples):
    model = Transformer(params = params, env=equation_env, samples=samples)
    model.to(params.device) 
    generations_ref, gen_len_ref = respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None) 
    sequence_ref = generations_ref[0][:gen_len_ref-1].tolist()

    rl_env = RLEnv(
        params = params,
        samples = samples,
        equation_env = equation_env,
        model = model)

    dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty = params.beam_length_penalty,
            train_value_mode=params.train_value,
            debug=params.debug)

    agent = UCT(
        action_space=[], # this will not be used as we have a default policy
        gamma=1., # no discounting
        ucb_constant=1.,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base)

    agent.display()
    if params.sample_only:
        # a bit hacky, should set a large rollout number so all programs are saved in samples json file
        horizon = 1
    else:
        horizon = 200    
        
    done = False
    s = rl_env.state
    ret_all = []
    for t in range(horizon):
        if len(s) >= params.horizon:
            print(f'Cannot process programs longer than {params.horizon}. Stop here.')
            break
        if done:
            break
        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)
        if t ==0:
            real_root = agent.root
        if params.debug:
            # print the current tree
            print('tree:')
            # print_tree(agent.root, equation_env.equation_id2word)
            ret = convert_to_json(agent.root, rl_env, act)
            ret_all.append(ret)
            
            with open("tree.json", "w") as outfile:
                json.dump(ret_all, outfile)

            print('took action:')
            print(repr(equation_env.equation_id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)

    y_mcts , mcts_str , mcts_tree = pred_for_sample_no_refine(model, equation_env, s ,samples['x_to_fit'])
    y_mcts_refine , mcts_str_refine, mcts_tree_refine = refine_for_sample(params, model, equation_env, s,  samples['x_to_fit'], samples['y_to_fit'])

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    for op,replace_op in replace_ops.items():
        mcts_str = mcts_str.replace(op,replace_op)

    mcts_eq = sp.parse_expr(mcts_str)
    mcts_eq_refine = sp.parse_expr(mcts_str_refine)
    expr = mcts_eq_refine 

    print("\nTPSR+E2E Equation: ", mcts_eq)
    print("\nTPSR+E2E Equation after Refine: ", mcts_eq_refine)
    print('#'*40)
    return expr 

def main_nesymres(case,params,eq_setting,cfg,samples,X,y):
    ## Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

    params_fit = FitParams(word2id=eq_setting["word2id"], 
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                una_ops=eq_setting["una_ops"], 
                                bin_ops=eq_setting["bin_ops"], 
                                total_variables=list(eq_setting["total_variables"]),  
                                total_coefficients=list(eq_setting["total_coefficients"]),
                                rewrite_functions=list(eq_setting["rewrite_functions"]),
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

    weights_path = "./TPSR/nesymres/weights/100M.ckpt"

    ## Load architecture, set into eval mode, and pass the config parameters
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()

    fitfunc = partial(model.fitfunc,cfg_params=params_fit)

    output_ref = fitfunc(X,y) 

    ### MCTS 
    rl_env = RLEnv(
        params = params,
        samples = samples,
        model = model,
        cfg_params=params_fit)

    ## Get self.encoded in the model to use for Sequence generation from given states
    model.to_encode(X,y, params_fit)

    dp = NesymresHeuristic(
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty = params.beam_length_penalty,
            cfg_params = params_fit,
            train_value_mode=params.train_value,
            debug=params.debug)

    agent = UCT(
        action_space=[],
        gamma=1., 
        ucb_constant=1.,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True
    )

    agent.display()

    if params.sample_only:
        horizon = 1
    else:
        horizon = 200

    done = False
    s = rl_env.state
    for t in range(horizon):
        if len(s) >= params.horizon:
            print(f'Cannot process programs longer than {params.horizon}. Stop here.')
            break

        if done:
            break

        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)

        if params.debug:
            # print the current tree
            print('tree:')
            print_tree(agent.root, params_fit.id2word)

            print('took action:')
            print(repr(params_fit.id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)

    loss_bfgs_mcts , reward_mcts , pred_str = compute_reward_nesymres(model.X, model.y, s, params_fit)
    print("TPSR+NeSymReS Equation: ", pred_str)
    expr = sp.parse_expr(pred_str)
    return expr

def eval_expr(expr:sp.Expr, X:np.ndarray) -> np.ndarray:
    """
    Evaluates a sympy expression for the given samples.

    Parameters:
        expr: sympy expression
        X: numpy array of shape (n_samples, n_features), sample points

    Returns:
        numpy array of evaluated values, one for each row in X
    """
    x_symbs = [f'x_{i}' for i in range(X.shape[1])]
    exec_func = sp.lambdify(x_symbs, expr)
    pred = exec_func(*[X[:, i] for i in range(X.shape[1])])
    if isinstance(pred, numbers.Number):
        pred = pred*np.ones(X.shape[0])
    return pred
