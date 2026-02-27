import os
import json
import logging
import datetime
import warnings
import itertools
import time
import traceback
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor
from sklearn.model_selection import train_test_split

from utils import generate_problem, convert_sets_in_dict, load_json_objects_from_folder, snap_numbers_in_expr
import metrics 

# Regressor Interfaces
#from linear_regressor import LinReg
#from polynomial_regressor import PolyReg
#from pysr_regressor import PySR
#from operon_regressor import Operon
#from tpsr_regressor import TransformerTPSR
#from gplearn_regressor import GPlearn
#from dsr_regressor import DSR
#from udsr_regressor import uDSR
#from nesymres_regressor import TransformerBiggio
#from symformer_regressor import TransformerVastl
#from lalande_regressor import TransformerLalande
from e2e_regressor import TransformerKamienny

# Params
#regressor = LinReg() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = PolyReg(degree=4) # train_domain_bounds (-10, 10), max_num_inputs = 10 #
#regressor = PySR() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = Operon() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = TransformerTPSR() # train_domain_bounds (-3, 3), max_num_inputs = 10
#regressor = GPlearn() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = DSR() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = PolyReg(degree=4, use_stand=False) # train_domain_bounds (0.1, 10), max_num_inputs = 6 # 
#regressor = uDSR() # train_domain_bounds (-10, 10), max_num_inputs = 10
#regressor = TransformerBiggio() # train_domain_bounds (-10, 10), max_num_inputs = 3
#regressor = TransformerVastl() # train_domain_bounds (-5, 5), max_num_inputs = 2
#regressor = TransformerLalande() # train_domain_bounds (0.1, 10), max_num_inputs = 6
regressor = TransformerKamienny()# train_domain_bounds (-3, 3), max_num_inputs = 10

# --- SCRIPT CONFIGURATION ---
algorithm_version = "e2e_regressor"
train_domain_bounds = (-3,3)
max_num_inputs = 10 
SAVE_INTERVAL = 10  

# --- HELPER FUNCTIONS (No changes from previous optimized version) ---
def get_fallback(task):
    """Generates a fallback result dictionary for a failed task."""
    try:
        expr_true = sp.parse_expr(str(task["expression"]))
    except:
        expr_true = parse_expr(str(task["expression"]), transformations=(standard_transformations + (convert_xor,)))

    return {
        'complexity': sp.count_ops(expr_true),
        'dimensionality': len(task["support"]),
        'recovery': False, 'tree_edit_distance': 1.0, 'jaccard_idx': 0.0,
        'ood_r2_score': 0.0, 'train_r2_score': 0.0, 'id_r2_score': 0.0,
        'expr_tree_ops': -1.0, 'time': -1.0, 'is_valid': False
    }

def generate_param_combinations(default_params, experiment_params):
    """Generates all combinations of parameters for an experiment."""
    combined_params = {**default_params, **experiment_params}
    keys = combined_params.keys()
    values = [v if isinstance(v, list) else [v] for v in combined_params.values()]
    
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def create_metrics_dict(algorithm_version, dataset_name, experiment_name, task, param_dict, benchmark_metrics, elapsed_time, expr_true, expr_est, X=None):
    """Builds the final, consolidated dictionary of all metrics for a single task run."""
    num_inputs = X.shape[1] if X is not None and hasattr(X, "shape") else len(task.get("support", []))
    return {
        "algorithm_version": algorithm_version, "dataset_name": dataset_name, "experiment_name": experiment_name,
        "task_id": task["task_id"], "num_inputs": num_inputs, "inference_time": elapsed_time,
        "expr_true": str(expr_true), "expr_est": str(expr_est),
        **benchmark_metrics, **param_dict
    }

def handle_task_error(logger, e, context_msg, task, **kwargs):
    """Logs an error, gets fallback metrics, and creates the final metrics dict for the failed task."""
    logger.error(f"{context_msg} for task {task.get('task_id', 'N/A')}. Error: {e}")
    logger.exception("ERROR TRACE:")
    fallback_metrics = get_fallback(task)
    kwargs.setdefault('expr_true', task.get('expression', ''))
    kwargs.setdefault('expr_est', None)
    kwargs.setdefault('elapsed_time', -1.0)
    return create_metrics_dict(benchmark_metrics=fallback_metrics, task=task, **kwargs)

def save_results_to_json(filepath, data_dict):
    """Saves the entire results dictionary to a JSON file."""
    with open(filepath, "w") as f:
        # Assuming convert_sets_in_dict handles non-serializable types like sets
        json.dump(convert_sets_in_dict(data_dict), f, indent=4)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. --- SETUP LOGGING ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"./_logs/benchmark_{algorithm_version}.log", mode='w')])
    logger = logging.getLogger()

    # 2. --- LOAD DATASETS ---
    dataset_folder_paths = {"feynman": "../datasets/Feynman", "strogatz": "../datasets/Strogatz"}
    benchmark_datasets = {name: load_json_objects_from_folder(path) for name, path in dataset_folder_paths.items()}

    # 3. --- SETUP RESULTS DICTIONARY (Optimized for standard JSON) ---
    results_dict_file_path = f"./_results/benchmark_result_{algorithm_version}.json" # <-- CHANGE: File extension is .json
    overwrite = True
    result_dict = {}

    if os.path.exists(results_dict_file_path) and not overwrite:
        logger.info(f"Loading results dict from {results_dict_file_path}")
        with open(results_dict_file_path, "r") as f:
            result_dict = json.load(f)
        # Ensure essential keys exist
        result_dict.setdefault("finished_tasks", {})
        result_dict.setdefault("results", [])
    else:
        logger.info("Creating a new results dictionary.")
        result_dict = {
            "experiment_date": str(datetime.datetime.now()),
            "algorithm_version": algorithm_version,
            "finished_tasks": {}, 
            "results": [],
        }

    # 4. --- DEFINE EXPERIMENT CONFIGURATION ---
    warnings.filterwarnings("ignore")
    default_values = {
        "train_domain_bounds": train_domain_bounds, "total_domain_bounds": (-100, 100), "noise_level": 0,
        "sampling_distribution": "uniform", "domain_box": "in_domain", "num_train_points": 255, 
        "test_set_ratio": 0.2, "seed": 42,
    }
    experiment_config = {
        "performance_experiments": [
            {"domain_box": ["in_domain", "within_domain", "out_of_domain"], "sampling_distribution": ["uniform", "diverse", "normal"], 
             "noise_level": [0], "num_train_points": [255]}, #320
        ],
        "ood_shift_experiments": [
            {"domain_box": ["out_of_domain"], "sampling_distribution": ["uniform"], "train_domain_bounds": [(-10, 10), (-20, 20), (-30, 30), (-40, 40), (-50, 50), (-60, 60), (-70, 70), (-80, 80), (-90, 90)],
             "noise_level": [0], "num_train_points": [255]}, 
        ]
    }

    # 5. --- RUN EXPERIMENTS ---
    try:
        for experiment_name, criteria_list in experiment_config.items():
            logger.info(f"--- Starting Experiment: {experiment_name} ---")

            for criteria in criteria_list:
                for param_dict in generate_param_combinations(default_values, criteria):
                    
                    ablation_name = f'{algorithm_version}_{param_dict["domain_box"]}_{param_dict["sampling_distribution"]}'

                    for dataset_name, task_list in benchmark_datasets.items():
                        logger.info(f"Evaluating Dataset: {dataset_name} with Parameters: {param_dict}")

                        for cnt, task in enumerate(task_list):
                            task_id = f"{experiment_name}_{ablation_name}_{dataset_name}_{task['task_id']}_{str(param_dict['train_domain_bounds'])}"

                            if task_id in result_dict["finished_tasks"]:
                                logger.info(f"Skipping already finished task {cnt+1}/{len(task_list)}: {task_id}")
                                continue
                            
                            logger.info(f"Running task {cnt+1}/{len(task_list)} of dataset {dataset_name}: {task['task_id']}")

                            X, y, expr_true, expr_est = None, None, None, None
                            elapsed_time = -1.0
                            all_metrics = {}

                            try:
                                # A: Data Generation, B: Model Fitting, C: Metrics Computation...
                                expr_true, X_train, y_train, X_test_id, y_test_id, X_test_ood, y_test_ood = generate_problem(
                                    task=task, domain_box=param_dict["domain_box"], 
                                    sampling_distribution=param_dict["sampling_distribution"], 
                                    num_train_points=param_dict["num_train_points"], 
                                    train_domain_bounds=param_dict["train_domain_bounds"], 
                                    total_domain_bounds=param_dict["total_domain_bounds"], 
                                    seed=param_dict["seed"], 
                                    test_set_ratio=param_dict["test_set_ratio"], 
                                    noise_level=param_dict["noise_level"]
                                )

                                if X_train.shape[1] > max_num_inputs:
                                    logger.warning(f"Skipping task due to high dimensionality ({X_train.shape[1]} > {max_num_inputs})")
                                    benchmark_metrics = get_fallback(task)
                                else:      

                                    start_time = time.time()
                                    regressor.fit(X_train, y_train)
                                    elapsed_time = time.time() - start_time
                                    
                                    expr_est = regressor.model()
                                    pred_train_id, pred_test_id, pred_test_ood = regressor.predict(X_train), regressor.predict(X_test_id), regressor.predict(X_test_ood)

                                    # snap numbers in expression for symbolic evaluation
                                    expr_true = snap_numbers_in_expr(expr_true)
                                    expr_est = snap_numbers_in_expr(expr_est)

                                    # flatten ys to 1D array for eval interface
                                    y_train, y_test_id, y_test_ood = y_train.flatten(), y_test_id.flatten(), y_test_ood.flatten() 

                                    benchmark_metrics = {
                                        'complexity': sp.count_ops(expr_true), 'dimensionality': X_train.shape[1],
                                        'recovery': metrics.recovery(expr_true, expr_est, X_train, symbolic=True),
                                        'tree_edit_distance': metrics.tree_edit_distance(expr_true, expr_est, norm=expr_true),
                                        'jaccard_idx': metrics.jaccard_index(expr_true, expr_est, X=X_train, symbolic=False),
                                        'ood_r2_score': metrics.r_squared_score(y_test_ood, pred_test_ood, y_hat=np.mean(y_test_ood)),
                                        'train_r2_score': metrics.r_squared_score(y_train, pred_train_id, y_hat=np.mean(y_train)),
                                        'id_r2_score': metrics.r_squared_score(y_test_id, pred_test_id, y_hat=np.mean(y_test_id)),
                                        'expr_tree_ops': sp.count_ops(expr_est), 'time': elapsed_time, 'is_valid': True
                                    }
                                
                                all_metrics = create_metrics_dict(
                                    algorithm_version=algorithm_version, dataset_name=dataset_name, experiment_name=experiment_name,
                                    task=task, param_dict=param_dict, benchmark_metrics=benchmark_metrics,
                                    elapsed_time=elapsed_time, expr_true=expr_true, expr_est=expr_est, X=X_train
                                )

                            except Exception as e:
                                context_msg = f"An unexpected error occurred during processing of experiment {experiment_name}"
                                all_metrics = handle_task_error(logger, e, context_msg, task=task,
                                    algorithm_version=algorithm_version, dataset_name=dataset_name, 
                                    experiment_name=experiment_name, param_dict=param_dict, X=X)

                            # E: Add result to in-memory dictionary
                            result_dict["results"].append(all_metrics)
                            result_dict["finished_tasks"][task_id] = True

                            # --- CHANGE: PERIODIC SAVE ---
                            # Save progress periodically based on the SAVE_INTERVAL
                            if (cnt + 1) % SAVE_INTERVAL == 0:
                                logger.info(f"Periodic save triggered. Saving {len(result_dict['results'])} results to disk...")
                                save_results_to_json(results_dict_file_path, result_dict)
    finally:
        # 6. --- FINAL SAVE ---
        # Always save the final results when the script finishes or encounters a critical error
        logger.info(f"--- Benchmark Run Finished or Interrupted. Saving final results to: {results_dict_file_path} ---")
        save_results_to_json(results_dict_file_path, result_dict)
        logger.info("Final save complete.")

