'''
Evaluation metrics used for the benchmark
'''
import numpy as np
import sympy as sp
from sympy import preorder_traversal, simplify, srepr
from sklearn.metrics import pairwise_distances

try:
    import src.zss as zss
    import src.utils as utils
except: 
    import zss
    import utils

def create_metrics_dict() -> dict:
    # create metrics dict value = tuple (evaluation function, fallback value)
    symbolic_metrics = {
        # norm = anything
        'recovery' : lambda X, y, expr_true, expr_est, pred, norm=None: recovery(expr_true, expr_est, X),
        'tree_edit_distance' : lambda X, y, expr_true, expr_est, pred, norm=None: tree_edit_distance(expr_true, expr_est, norm),
    }
    numeric_metrics = {
        # norm = y_hat
        'r_squared' : lambda X, y, expr_true, expr_est, pred, norm=None: r_squared_score(y, pred, norm),
        'mae' : lambda X, y, expr_true, expr_est, pred, norm=None: mean_absolute_error(y, pred, norm),
        'mse' : lambda X, y, expr_true, expr_est, pred, norm=None: mean_squared_error(y, pred, norm),
        'rmse' : lambda X, y, expr_true, expr_est, pred, norm=None: root_mean_squared_error(y, pred, norm),
        'medl' : lambda X, y, expr_true, expr_est, pred, norm=None: mean_error_description_length(y, pred, norm),
    }
    complexity_metrics = {
        # norm = expr_true
        'treeops' : lambda X, y, expr_true, expr_est, pred, norm=None: expr_tree_ops(expr_est, norm),
        'treebits' : lambda X, y, expr_true, expr_est, pred, norm=None: expr_tree_bits(expr_est, norm),
        'treesize' : lambda X, y, expr_true, expr_est, pred, norm=None: expr_tree_size(expr_est, norm),
    }
    tradeoff_metrics = {
        # norm = (y_hat, expr_true)
        'pareto_score' : lambda X, y, expr_true, expr_est, pred, norm=None: pareto_score(y, pred, expr_est, norm),
    }
    eval_metrics = {
        'symbolic' : symbolic_metrics,
        'numeric' : numeric_metrics,
        'complexity' : complexity_metrics,
        'tradeoff' : tradeoff_metrics,
    }
    return eval_metrics


####################################################################################################
# Symbolic Evaluation metrics
####################################################################################################

def recovery(expr_true:sp.Expr, expr_est:sp.Expr, X:np.ndarray, symbolic:bool = True) -> bool:
    '''
    Checks symbolic equivalence of expressions.

    Params:
        expr_est... sympy expression
        expr_true... sympy expression
        X... samples at which we know that expr must evaluate
        symbolic... whether to check symbolic equivalence or numerical equivalence
    
    Returns:
        True if both expressions are equivalent.
    
    '''
    if symbolic:
        const_func = utils.is_const_symbolic
    else:
        const_func = lambda expr: utils.is_const_numeric(expr, X)

    # 1. difference reduces to constant
    expr_diff = expr_true - expr_est
    if const_func(expr_diff):
        return True
    else:
        # 2. ratio reduces to constant which is not 0

        # Check all cases: 
        # a) expr_true == expr_est == 0 -> return True -> this case should be handled by the difference check above
        # b) expr_true == 0 and expr_est != 0 -> return False 
        # c) expr_true != 0 and expr_est == 0 -> return False  
        # d) expr_true != 0 and expr_est != 0 -> check if expr_ratio is a constant which is not zero -> return True or False
        
        #if expr_true == 0:
        #    return False

        #if expr_est == 0:
        #    return False

        try:
            expr_ratio = expr_true/expr_est 
        except Exception:
            return False

        if expr_ratio is not None and len(expr_ratio.free_symbols) == 0:
            # check it's not zero
            try:
                val = float(sp.N(expr_ratio))
                if not np.isclose(val, 0.0, atol=1e-8, rtol=1e-6):
                    return True
                else:
                    return False
            except:
                return False
        else:
            # check if the expression ratio is a constant != 0
            return utils.is_constant_non_zero(expr_ratio)

def subtrees_structural(expr, include_atoms=True):
    """Return a set of SymPy Basic nodes (structural)."""
    if include_atoms:
        return set(preorder_traversal(expr))
    else:
        return set(x for x in preorder_traversal(expr) if x.args)  # exclude atoms

def jaccard_structural(expr_a, expr_b, include_atoms=True):
    A = subtrees_structural(expr_a, include_atoms)
    B = subtrees_structural(expr_b, include_atoms)
    if not (A or B):
        return 1.0  # both empty -> define J=1
    inter = A & B
    union = A | B
    return float(len(inter)) / float(len(union))

def eval_all_intermediates_numpy(expr: sp.Expr, X: np.ndarray):
    """
    Evaluates all subexpressions of `expr` using NumPy only,
    reusing intermediate results.

    Parameters:
        expr: sympy expression
        X: numpy array of shape (n_samples, n_features)

    Returns:
        Matrix of shape (n_subexprs, n_samples), each row corresponds to a subexpression
    """
    x_symbs = [sp.Symbol(f"x_{i}") for i in range(X.shape[1])]
    cache = {}
    n_samples = X.shape[0]

    def eval_node(e):
        if e in cache:
            return cache[e]

        if e.is_Symbol:  # variable
            idx = int(str(e).split("_")[1])
            val = X[:, idx]

        elif e.is_Number:  # constant
            val = np.full(n_samples, float(e))

        else:
            # recursively evaluate children
            child_vals = [eval_node(arg) for arg in e.args]

            # map SymPy operations to NumPy
            if isinstance(e, sp.Add):
                val = np.sum(child_vals, axis=0)
            elif isinstance(e, sp.Mul):
                val = np.prod(child_vals, axis=0)
            elif isinstance(e, sp.Pow):
                base, exp = child_vals
                val = np.power(base, exp)
            elif isinstance(e, sp.sin):
                val = np.sin(child_vals[0])
            elif isinstance(e, sp.cos):
                val = np.cos(child_vals[0])
            elif isinstance(e, sp.exp):
                val = np.exp(child_vals[0])
            elif isinstance(e, sp.log):
                val = np.log(child_vals[0])
            elif isinstance(e, sp.tan):
                val = np.tan(child_vals[0])
            elif isinstance(e, sp.cot):
                # cot(x) = 1 / tan(x)
                # This will raise a RuntimeWarning and produce `inf` where tan(x) is 0.
                val = 1.0 / np.tan(child_vals[0])
            else:
                # fallback to lambdify if operation not covered
                f = sp.lambdify(x_symbs, e, modules="numpy")
                val = f(*[X[:, i] for i in range(X.shape[1])])

        # Ensure the result is always a vector of size n_samples.             #
        # This handles cases where lambdify (or another operation) returns a  #
        # scalar for a constant subexpression.                                #
        if np.isscalar(val):
            val = np.full(n_samples, val, dtype=float)

        cache[e] = val
        return val

    # force evaluation of root
    eval_node(expr)

    # return results in evaluation order
    # Note: Use np.vstack for clarity, it does the same as row_stack here.
    return np.vstack([cache[sub] for sub in sp.preorder_traversal(expr)])

def jaccard_index(expr_true:sp.Expr, expr_est:sp.Expr, X:np.ndarray, symbolic:bool = True) -> float:
    '''
    Calculates the jaccard index of two sympy expressions.

    Params:
        expr_true... sympy expression
        expr_est... sympy expression
 
    Returns:
        Jaccard index of the two expressions 
    '''
    # check if expressions are sympy expressions
    try:
        if isinstance(expr_true, str):
            print(f"Warning: expr_true is a string. Converting to sympy expression.")
            expr_true = sp.sympify(expr_true)
        if isinstance(expr_est, str):
            print(f"Warning: expr_est is a string. Converting to sympy expression.")
            expr_est = sp.sympify(expr_est)
    except:
        print(f"Warning: expr_true or expr_est is not a sympy expression. Skipping Jaccard index calculation.")
        return 0

    if symbolic: 
        tmp_expr1 = utils.replace_numbers_in_expr_with_placeholders(expr_true)
        tmp_expr2 = utils.replace_numbers_in_expr_with_placeholders(expr_est)
        alt1 = utils.get_alternatives(tmp_expr1)
        alt2 = utils.get_alternatives(tmp_expr2)
        M1 = [set([str(subexpr) for subexpr in utils.get_subexprs_sympy(tmp_expr1)]) for tmp_expr1 in alt1]
        M2 = [set([str(subexpr) for subexpr in utils.get_subexprs_sympy(tmp_expr2)]) for tmp_expr2 in alt2]

        try: 
            jaccard_idx = max([len(S1&S2)/len(S1|S2) for S1 in M1 for S2 in M2])
        except:
            # try fallback 
            try:
                jaccard_idx = jaccard_structural(expr_true, expr_est)
            except:
                print(f"Warning: Jaccard index calculation failed. Returning 0.")
                jaccard_idx = 0
    else:
        M1 = eval_all_intermediates_numpy(expr_true, X)
        M2 = eval_all_intermediates_numpy(expr_est, X)

        # Sanitize the matrices to remove NaNs and infinities before calculating distances.
        M1 = np.nan_to_num(M1)
        M2 = np.nan_to_num(M2)

        D = pairwise_distances(M1, M2, metric = 'l1')

        mask = np.isclose(D, 0)
        intersect = np.sum(np.any(mask, axis=1))
        union = len(M1) + len(M2) - intersect # Union = |A| + |B| - |A ∩ B|
        if union == 0:
            return 1.0 if intersect == 0 else 0.0 # Handle division by zero
        
        jaccard_idx = intersect/union

    return jaccard_idx

def tree_edit_distance(expr_true:sp.Expr, expr_est:sp.Expr, norm:float = None) -> float:
    '''
    Computes the distance between two equation trees.

    Params:
        expr_true... sympy expression
        expr_est... sympy expression
        normalize... whether to normalize the distance by the number of nodes in the ground truth expression
        y_hat... average value of training data (not used in this function, but included for compatibility with other metrics)
        
    Returns:
        (normalized) tree-edit-distance between the two expressions
    '''

    tree1 = utils.expr2tree(expr_true)
    tree2 = utils.expr2tree(expr_est)
    edit_dist = zss.simple_distance(tree1, tree2)
    if norm is None:
        return edit_dist

    num_gt_nodes = utils.count_nodes(tree1)
    return min([edit_dist, num_gt_nodes]) / num_gt_nodes

####################################################################################################
# Numerical Evaluation metrics
####################################################################################################

def r_squared_score(y:np.ndarray, pred:np.ndarray, y_hat:float = None) -> float:
    '''
    Calculates coefficient of determination between true values and prediction.

    Params:
        y... true dependent values
        pred... predicted dependent values
        y_hat... average value of training data

    Returns:
        R2 Score in (-inf, 1), 1 is best
        or 
        R2+ Score in (0, 1) if y_hat is not None, 1 is best

    '''
    
    den = np.sum((y - np.mean(y))**2)
    num = np.sum((y - pred)**2)
    r2 = 1 - num/den

    if y_hat is not None:
        default_value = 0
        r2 = max(0, r2)
    else:
        default_value = -1e5

    if not (np.isfinite(r2) and np.isreal(r2)):
        return default_value
    return r2

def mean_absolute_error(y:np.ndarray, pred:np.ndarray, y_hat:float = None) -> float:
    '''
    Calculates mean absolute error between true values and prediction.

    Params:
        y... true dependent values
        pred... predicted dependent values
        y_hat... average value of training data

    Returns:
        MAE in [0, inf), 0 is best
        or 
        MAE+ in (0, 1) if y_hat is not None, 1 is best
    '''
    score = np.mean(np.abs(y - pred))

    if y_hat is not None:
        default_value = 1
        score = min(1, score/(np.mean(np.abs(y - y_hat))))
    else:
        default_value = 1e5
    if not (np.isfinite(score) and np.isreal(score)):
        return default_value
    return score

def mean_squared_error(y:np.ndarray, pred:np.ndarray, y_hat:float = None) -> float:
    '''
    Calculates mean squared error between true values and prediction.

    Params:
        y... true dependent values
        pred... predicted dependent values
        y_hat... average value of training data

    Returns:
        MSE in [0, inf), 0 is best
        or 
        MSE+ in (0, 1) if y_hat is not None, 1 is best
    '''
    score = np.mean((y - pred)**2)

    if y_hat is not None:
        default_value = 1
        score = min(1, score/(np.mean((y - y_hat)**2)))
    else:
        default_value = 1e5
    if not (np.isfinite(score) and np.isreal(score)):
        return default_value
    return score

def root_mean_squared_error(y:np.ndarray, pred:np.ndarray, y_hat:float = None) -> float:
    '''
    Calculates root mean squared error between true values and prediction.

    Params:
        y... true dependent values
        pred... predicted dependent values
        y_hat... average value of training data

    Returns:
        RMSE in [0, inf), 0 is best
        or 
        RMSE+ in (0, 1) if y_hat is not None, 1 is best
    '''
    score = np.sqrt(np.mean((y - pred)**2))

    if y_hat is not None:
        default_value = 1
        score = min(1, score/(np.sqrt(np.mean((y - y_hat)**2))))
    else:
        default_value = 1e5
    if not (np.isfinite(score) and np.isreal(score)):
        return default_value
    return score

def mean_error_description_length(y:np.ndarray, pred:np.ndarray, y_hat:float = None) -> float:
    '''
    Calculates number of bits needed to describe the error between true values and prediction.

    Params:
        y... true dependent values
        pred... predicted dependent values
        y_hat... average value of training data

    Returns:
        MEDL in [0, inf), 0 is best
        or 
        MEDL+ in (0, 1) if y_hat is not None, 1 is best
    '''

    medl = np.mean(np.log2(1+abs(pred-y)*2**30))
    if y_hat is not None:
        default_value = 1
        medl = min(1, medl/(np.mean(np.log2(1+abs(y-y_hat)*2**30))))
    else:
        default_value = 1000000
    if not (np.isfinite(medl) and np.isreal(medl)):
        return default_value
    return medl

####################################################################################################
# Complexity metrics
####################################################################################################

def expr_tree_ops(expr:sp.Expr, expr_true:sp.Expr = None) -> int:
    '''
    Calculates number of operations in expression tree according to sympy.

    Params:
        expr... sympy expression
        expr_true... true expression used for normalization

    Returns:
        number of operations in [0, inf)
        or
        number of operations in (0, 1) if expr_true is not None, 0 is best
    '''
    ops_expr = sp.count_ops(expr)
    if expr_true is not None:
        default_value = 1
        ops_true = sp.count_ops(expr_true)
        ops_expr = min(ops_expr, 2*ops_true)/(2*ops_true)
    else:
        default_value = 1000000
    if not (np.isfinite(ops_expr) and np.isreal(ops_expr) and ops_expr >= 0):
        return default_value
    return ops_expr

def expr_tree_bits(expr:sp.Expr, expr_true:sp.Expr = None) -> float:
    '''
    Calculates number of bits needed to represent expression.

    Params:
        expr... sympy expression
        expr_true... true expression used for normalization

    Returns:
        number of bits in [0, inf)
        or
        number of bits in (0, 1) if expr_true is not None, 0 is best
    '''

    expr = sp.parse_expr(str(expr))
    compl = 0
    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    numbers_expr = [subexpression for subexpression in sp.preorder_traversal(expr) if is_atomic_number(subexpression)]
    for j in numbers_expr:
        if j.is_real:
            compl = compl + utils.get_number_complexity(float(j), use_number_snapping=True)
        else:
            print(f"Warning: complex number encountered in expression: {expr}, number: {j}. Workaround: use abs(number) instead.")
            compl += utils.get_number_complexity(float(abs(j)), use_number_snapping=True)

    n_variables = len(expr.free_symbols)
    n_operations = expr_tree_ops(expr)
    if n_operations!=0 or n_variables!=0:
        compl += (n_variables+n_operations)*np.log2((n_variables+n_operations))

    if expr_true is not None:
        default_value = 1
        compl_true = expr_tree_bits(expr_true)
        compl = min(compl, 2*compl_true)/(2*compl_true)
    else:
        default_value = 1000000
    if not (np.isfinite(compl) and np.isreal(compl) and compl >= 0):
        return default_value
    return compl

def expr_tree_size(expr:sp.Expr, expr_true:sp.Expr = None) -> int:
    '''
    Calculates number of nodes in expression tree according to sympy.

    Params:
        expr... sympy expression
        expr_true... true expression used for normalization

    Returns:
        number of nodes in [0, inf)
        or
        number of nodes in (0, 1) if expr_true is not None, 0 is best
    '''
    treesize = len(list(sp.preorder_traversal(expr)))

    if expr_true is not None:
        default_value = 1
        treesize_true = len(list(sp.preorder_traversal(expr_true)))
        treesize = min(treesize, 2*treesize_true)/(2*treesize_true)
    else:
        default_value = 1000000
    if not (np.isfinite(treesize) and np.isreal(treesize) and treesize >= 0):
        return default_value

