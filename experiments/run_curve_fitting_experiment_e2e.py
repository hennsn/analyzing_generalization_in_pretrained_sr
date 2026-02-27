import pickle
import sympy as sp
import numpy as np
from sklearn.metrics import r2_score
import warnings
from scipy.optimize import minimize

from regressors.e2e_regressor import TransformerKamienny

def replace_constants(expr):
    # Find all numbers in the expression
    numbers = sorted(expr.atoms(sp.Number), key=lambda x: str(x))
    # Create mapping c_0, c_1, ...
    replacements = {num: sp.Symbol(f'c_{i}') for i, num in enumerate(numbers)}
    # Replace in expression
    new_expr = expr.xreplace(replacements)
    return new_expr, np.array(numbers).astype(float)

def expr2np(expr):
    """
    Creates a numpy string that can be evaluated using 'eval'.

    Params:
        expr... sympy expression
        d... dimensionality of input

    Returns:
        expression string with numpy operations
    """
    repl_dict = {
        'exp' : 'np.exp',
        '^' : '**',
        'asin' : 'np.arcsin',
        'acos' : 'np.arccos',
        'arctan' : 'np.arctan',
        'sin' : 'np.sin',
        'cos' : 'np.cos',
        'log' : 'np.log',
        'ln' : 'np.log',
        'tan' : 'np.tan',
        'cot' : '1/np.tan',
        'sqrt' : 'np.sqrt',
        'pi' : 'np.pi',
        'E' : 'np.e',
        'Abs' : 'np.abs',
    }
    
    expr_str = str(expr)
    for s in sorted(list(repl_dict.keys())):
        expr_str = expr_str.replace(s, repl_dict[s])
    expr_str = expr_str.replace('arcnp.', 'arc')
        
    # 1. replace constants
    idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols if 'c_' in str(s)], reverse = True)
    for idx in idxs:
         expr_str = expr_str.replace(f'c_{idx}', f'c[{idx}]')
    # 1. replace variables
    idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols if 'x_' in str(s)], reverse = True)
    for idx in idxs:
         expr_str = expr_str.replace(f'x_{idx}', f'X[:, {idx}]')

    return expr_str


################################################################################
# E2E Curve Fitting Experiment
################################################################################
def run_e2e_experiment():
    with open('./data/tasks.p', 'rb') as handle:
        task_dict = pickle.load(handle)
    problem_names = list(task_dict.keys())

    for d, start_idx in [(2, 7), (3, 3), (4, 2)]:
        all_idxs = [i for i in range(len(problem_names)) if task_dict[problem_names[i]]['X'].shape[1] == d]
        results = {'base_expression' : None, 'scores' : [], 'consts' : [], 'names' : []}
        
        regr = TransformerKamienny(random_state = 0) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # fit on first problem
            idx = all_idxs[start_idx]
            problem_name = problem_names[idx]
            X, y, expr = task_dict[problem_name]['X'], task_dict[problem_name]['y'][:, 0], task_dict[problem_name]['expr'][0]
            
            regr.fit(X, y)
            base_expr = regr.model()
            base_expr, c = replace_constants(base_expr)
            n_constants = 0
            for s in base_expr.free_symbols:
                s_str = str(s)
                if 'c_' in s_str:
                    idx = int(s_str.split('_')[-1])
                    n_constants = max(n_constants, idx+1)
            np.random.seed(0)
            C = np.random.rand(50000, n_constants)*2-1 # constants
            np_str = expr2np(base_expr)
        
            results['names'].append(problem_name)
            results['base_expression'] = str(base_expr)
            results['consts'].append(c.copy())
            pred = eval(np_str)
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            results['scores'].append(r2_score(y, pred))
            
            print(f'Index: {idx}, Problem: {problem_name}, Expression: {expr}, Score: {results["scores"][-1]}')
            
            # optimize params on other problems
            for idx in [i for i in all_idxs if i!=all_idxs[start_idx]]:
                problem_name = problem_names[idx]
                X, y, expr = task_dict[problem_name]['X'], task_dict[problem_name]['y'][:, 0], task_dict[problem_name]['expr'][0]
            
                scores = []
                for c in C:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred = eval(np_str)
                        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                        score = r2_score(y, pred)
                        scores.append(score)
                scores = np.array(scores)
                max_idx = np.argmax(scores)
                c = C[max_idx]
                # optimize rest with scipy
                def opt_func(c):
                    pred = eval(np_str)
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    return 1-r2_score(y, pred)
                res = minimize(fun = opt_func, x0 = c, method = 'BFGS')
                best_r2 = 1 - res['fun']
                best_c = res['x']
        
                results['names'].append(problem_name)
                results['consts'].append(best_c.copy())
                results['scores'].append(best_r2)
                
                print(f'Index: {idx}, Problem: {problem_name}, Expression: {expr}, Score: {results["scores"][-1]}')
        
        with open(f'./_results/e2e_{X.shape[1]}D.p', 'wb') as handle:
            pickle.dump(results, handle)


################################################################################
# Polynomial Experiment
################################################################################

import pickle
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sympy
import warnings

def run_polynomial_experiment():
    with open('./data/tasks.p', 'rb') as handle:
        task_dict = pickle.load(handle)
    problem_names = list(task_dict.keys())
    dims = [2, 3, 4]
    degrees = [1,2,3,4,5,6]

    for d in dims:
        # select indices that match dimension d
        all_idxs = [i for i in range(len(problem_names)) if task_dict[problem_names[i]]['X'].shape[1] == d]

        if len(all_idxs) == 0:
            print(f"No problems found for dimension {d}, skipping.")
            continue

        print(f"\n=== Dimension {d}: {len(all_idxs)} problems ===")

        for degree in degrees:
            print(f"\n-- degree = {degree} --")
            results = {'base_expression': None, 'scores': [], 'consts': [], 'names': []}

            # Build polynomial feature transformer for this dimension and degree.
            # include_bias=True => first feature is constant 1 (maps to c_0).
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            # prepare x variable symbols for sympy expression
            x_syms = [sympy.Symbol(f'x_{i}', real=True) for i in range(d)]

            # Fit per problem
            for idx in all_idxs:
                problem_name = problem_names[idx]
                X = task_dict[problem_name]['X']
                y = task_dict[problem_name]['y'][:, 0]  # as in your other code

                # compute polynomial feature matrix (n_samples, n_features)
                # fit_transform here only generates the feature matrix; we won't call poly.fit on X since it's deterministic
                X_poly = poly.fit_transform(X)

                # Fit linear regression with no intercept because bias is included as feature
                model = LinearRegression(fit_intercept=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_poly, y)

                coef = model.coef_.copy()  # length = n_features
                # compute predictions and r2
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = model.predict(X_poly)
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                r2 = float(r2_score(y, pred))

                # Append numeric results
                results['names'].append(problem_name)
                results['consts'].append(coef.copy())
                results['scores'].append(r2)

                print(f'Problem index {idx}: {problem_name} — r2={r2:.4f} (n_features={coef.size})')

            # Construct a sympy expression mapping c_i symbols to polynomial monomials
            # Get feature names / monomials from PolynomialFeatures
            try:
                feat_names = poly.get_feature_names_out([f'x_{i}' for i in range(d)])
            except AttributeError:
                # older sklearn: get_feature_names
                feat_names = poly.get_feature_names([f'x_{i}' for i in range(d)])

            # Create expression sum_i c_i * monomial_i
            monomials = []
            for j, fname in enumerate(feat_names):
                # fname is like '1' or 'x0' or 'x0 x1^2' depending on sklearn version,
                # but for get_feature_names_out with input names we get something like '1' or 'x0' or 'x0 x1'
                # we'll convert it into sympy monomials:
                # replace spaces with '*' and ensure x_i naming is consistent.
                # For some sklearn versions names look like 'x0 x1', for others 'x0^2' may appear.
                token = fname.replace(' ', '*').replace('^', '**')
                # ensure variable names match x_0, x_1, ...
                # we allow either 'x0' or 'x_0' in feature name and normalize to x_0
                for i in range(d):
                    token = token.replace(f'x{i}', f'x_{i}')
                if token == '1':
                    mon = sympy.Integer(1)
                else:
                    mon = sympy.sympify(token)
                monomials.append(mon)

            # create c symbols
            c_syms = [sympy.Symbol(f'c_{i}', real=True) for i in range(len(monomials))]
            expr = sum(c * m for c, m in zip(c_syms, monomials))

            # store base expression as string (so later sympy.sympify(...) works)
            results['base_expression'] = str(expr)

            # Save results to file named by (dimension, degree)
            out_name = f'./_results/poly_deg{degree}_{d}D.p'
            with open(out_name, 'wb') as handle:
                pickle.dump(results, handle)
            print(f"Saved results to {out_name} (n_problems={len(all_idxs)}, n_coeffs={len(monomials)})")

    print("\nAll done.")

run_polynomial_experiment()