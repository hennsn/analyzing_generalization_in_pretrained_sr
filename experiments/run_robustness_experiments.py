import numpy as np
import os
import pickle
import sympy
from datetime import datetime
import signal

# adjust import according to your project structure
from utils import eval_expr as evaluate_expression
from metrics import recovery as symbolic_equivalence

# uncomment target regressor
from e2e_regressor import TransformerKamienny
#from lalande_regressor import TransformerLalande
#from symformer_regressor import TransformerVastlCache #TransformerVastl
#from nesymres_regressor import TransformerBiggio
#from tpsr_regressor import TransformerTPSR

# Timeout handling
class TimeoutException(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, _handler)
timeout_seconds = 10

# Main function
def robustness_transformer(expr_true:sympy.Expr, X_init:np.ndarray, epsilons:list, n_tries:int, regr_class, regr_name:str, operation:str, save_path:str = None):
    '''
    Experiment to test the robustness of Transformers.

    Parameters
    ----------
    expr_true : sympy expression
        The true expression to be recovered.
    X_init : np.ndarray
        The initial input data from the pretraining domain.
    epsilons : list
        List of epsilon values to test the robustness.
    n_tries : int
        Number of trials to run for each epsilon.
    regr_class : class
        The regression class to be tested.
    regr_name : str
        The name of the regression algorithm.
    operation : str
        The type of operation to perform. 
        One of 'shift_input', 'scale_input', 'noise', 'distribution'.
    '''
    print(f'Running robustness experiment for operation: {operation}, regr_name: {regr_name}')
    
    a, b = np.min(X_init), np.max(X_init)
    x_range = b - a
    if save_path is None:
        print(f'No save path specified. Saving to ./_results/results_robustness_experiments_{operation}.p')
        save_path = f'./_results/results_robustness_experiments_{operation}.p'
    if not os.path.exists('./_results'):
        print('No results folder found. Creating folder ./_results')
        os.makedirs('./_results')

    if os.path.exists(save_path):
        print(f'Loading results from {save_path}')
        with open(save_path, 'rb') as handle:
            results = pickle.load(handle)
    else:    
        results = {}

    if regr_name not in results:
        results[regr_name] = {}

    for eps in epsilons:
        if eps not in results[regr_name]:
            results[regr_name][eps] = []
        k = len(results[regr_name][eps])
        if operation == 'shift_input':
            X = X_init + eps*(b-a)
            y = evaluate_expression(expr_true, X)
        elif operation == 'scale_input':
            X = X_init*eps*(b-a)
            y = evaluate_expression(expr_true, X)
        elif operation == 'noise':
            X = X_init
            y = evaluate_expression(expr_true, X)
            y += eps*np.sqrt(np.mean(y**2))
        elif operation == 'distribution':
            X = a + (b-a)*((X_init-a)/(b-a))**(np.exp(-eps))
            y = evaluate_expression(expr_true, X)
        else:
            raise ValueError('Operation not supported!')

        for i in range(k, n_tries):
            regr = regr_class(random_state = i)
            try:
                signal.alarm(timeout_seconds)
                regr.fit(X, y)
                rec = symbolic_equivalence(expr_true=expr_true, expr_est=regr.model(), X=None, symbolic=True)
            except TimeoutException:
                rec = False
                print(f'TimeoutException: Timeout after {timeout_seconds} s')
            except (KeyError, OverflowError, NameError, TypeError, ZeroDivisionError):
                rec = False
            except Exception as e:
                rec = False
                print(f'Exception: {e}')
            finally:
                signal.alarm(0)
            
            print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, Operation: {operation}, Regressor: {regr_name}, Eps: {eps}, Seed: {i}, Recovery: {rec}')
            results[regr_name][eps].append(rec)
            with open(save_path, 'wb') as handle:
                pickle.dump(results, handle)

if __name__ == '__main__':
    operations = ['shift_input', 'scale_input', 'noise', 'distribution']
    expr_true = sympy.sympify('x_0^2')
    epsilons = np.round(np.linspace(-5.0, 5.0, 57), 4)
    n_tries = 100

    # Transformer E2E
    if False:
        regr_class = TransformerKamienny
        regr_name = 'E2E'

        X_init = np.linspace(-1, 1, 100)
        X_init = X_init/np.std(X_init)
        X_init = X_init.reshape(-1, 1)

        for operation in ['shift_input', 'scale_input', 'noise', 'distribution']:
            save_path = f'./_results/results_robustness_experiments_{operation}.p'
            robustness_transformer(expr_true, X_init, epsilons, n_tries, regr_class, regr_name, save_path = save_path, operation=operation)

    # Transformer T4SR
    if False:
        regr_class = TransformerLalande
        regr_name = 'TF4SR'

        X_init = np.power(10.0, np.linspace(-1, 1, 100)).reshape(-1, 1)

        for operation in ['shift_input', 'scale_input', 'noise', 'distribution']:
            save_path = f'./_results/results_robustness_experiments_{operation}.p'
            robustness_transformer(expr_true, X_init, epsilons, n_tries, regr_class, regr_name, save_path = save_path, operation=operation)

    # Transformer SymFormer
    if False:
        regr_class = TransformerVastlCache
        regr_name = 'SymFormer'

        X_init = np.linspace(-5, 5, 100).reshape(-1, 1)

        for operation in ['shift_input', 'scale_input', 'noise', 'distribution']:
            save_path = f'./_results/results_robustness_experiments_{operation}.p'
            robustness_transformer(expr_true, X_init, epsilons, n_tries, regr_class, regr_name, save_path = save_path, operation=operation)

    # Transformer NeSymReS
    if False:
        regr_class = TransformerBiggio
        regr_name = 'NeSymReS'

        X_init = np.linspace(-10, 10, 100).reshape(-1, 1)

        for operation in ['shift_input', 'scale_input', 'noise', 'distribution']:
            save_path = f'./_results/results_robustness_experiments_{operation}.p'
            robustness_transformer(expr_true, X_init, epsilons, n_tries, regr_class, regr_name, save_path = save_path, operation=operation)

    # Transformer TPSR
    if False:
        regr_class = TransformerTPSR
        regr_name = 'TPSR'

        X_init = np.linspace(-1, 1, 100)
        X_init = X_init/np.std(X_init)
        X_init = X_init.reshape(-1, 1)

        for operation in ['shift_input', 'scale_input', 'noise', 'distribution']:
            save_path = f'./_results/results_robustness_experiments_{operation}.p'
            robustness_transformer(expr_true, X_init, epsilons, n_tries, regr_class, regr_name, save_path = save_path, operation=operation)