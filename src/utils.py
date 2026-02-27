#import stopit
import numpy as np
import scipy.stats
import sympy
import numbers
import traceback
import copy
import os
import json 
import re 
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

################################################################
# Utils
################################################################

def load_json_objects_from_folder(folder_path):
  """
  Loads all JSON files from a folder and returns a list of JSON objects.

  Args:
    folder_path: The path to the folder containing the JSON files.

  Returns:
    A list of JSON objects, or an empty list if no JSON files are found 
    or if an error occurs during processing.
  """

  json_objects = []
  try:
    for filename in os.listdir(folder_path):
      if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
          try:
            data = json.load(f)
            json_objects.append(data)
          except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {filename}: {e}")
  except FileNotFoundError:
    print(f"Folder not found: {folder_path}")
  except Exception as e:
      print(f"An error occured while processing files from {folder_path}: {e}")

  return json_objects

def convert_sets_in_dict(data):
    if isinstance(data, dict):
        # If it's a dictionary, iterate through its items
        return {key: convert_sets_in_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, apply the same to each item
        return [convert_sets_in_dict(item) for item in data]
    elif isinstance(data, set):
        # If it's a set, convert it to a list
        return list(data)
    else:
        # If it's neither a dict, list, nor set, return it as is
        return data

def snap_numbers_in_expr(expr, rel_tol=1e-9, abs_tol=1e-10, precision=5):
    expr_str = snap_numbers_in_expr_str(str(expr), rel_tol=1e-9, abs_tol=1e-10, precision=5)
    expr = sympy.sympify(expr_str)
    return expr

def snap_numbers_in_expr_str(expr_str, rel_tol=1e-9, abs_tol=1e-10, precision=5):
    """
    Snaps numbers in the expr_str.
    If a number is in exponential notation (e.g., 1.2e-5), it's replaced with "0".
    Otherwise, it's snapped using the snap_number function.
    """
    # A more robust regex to find various number formats:
    # - Optional sign (+ or -)
    # - Integers (e.g., 123)
    # - Floats with decimal point (e.g., 3.14, -0.5, .5, 5.)
    # - Scientific notation (e.g., 1e-5, -2.5E+10)
    # It captures the entire number string in group 1.
    # Using re.VERBOSE for clarity.
    number_pattern = r"""
    (?<![a-zA-Z0-9_.])           # Negative lookbehind: Not preceded by alphanumeric, dot, or underscore
    (                             # Start capturing group 1 (the whole number)
        [-+]?                     # Optional sign
        (?:                       # Non-capturing group for number structure:
            \d+\.\d*              # Digits, dot, optional digits (e.g., 3.14, 3.)
            | \.\d+               # OR: Dot, digits (e.g., .5)
            | \d+                 # OR: Integer (e.g., 123)
        )
        (?:[eE][-+]?\d+)?         # Optional: scientific notation part (e.g., e-5, E+10)
    )
    (?![a-zA-Z0-9_.])            # Negative lookahead: Not followed by alphanumeric, dot, or underscore
    """

    def replace_match(match):
        number_str = match.group(1)  # The entire matched number string

        # Check if the original number string uses exponential notation
        if 'e' in number_str.lower():  # '.lower()' handles 'e' and 'E'
            return "0"
        else:
            # If not exponential, proceed with the original snapping logic
            try:
                # CRITICAL: Pass the parameters from the outer function
                snapped_value = snap_number(number_str,
                                            rel_tol=rel_tol,
                                            abs_tol=abs_tol,
                                            precision=precision)
                return str(snapped_value) # Ensure the result is a string
            except ValueError:
                # If snap_number raises an error (e.g., for NaN, if it's configured to do so),
                # or if float(number_str) fails inside snap_number for some malformed string
                # (though the regex should prevent most malformed cases).
                return number_str  # Return the original number string

    # Replace all matches in the expression string
    snapped_expr_str = re.sub(number_pattern, replace_match, expr_str, flags=re.VERBOSE)
    return snapped_expr_str


def snap_number(number, rel_tol=1e-9, abs_tol=1e-10, precision=5):
    """
    Snaps a number to a short representation

    For less critical applications, rel_tol=1e-6 and abs_tol=1e-8 might suffice.
    """
    n = float(number)

    # Check for NaN
    if np.isnan(n):
        raise ValueError("NaN value encountered")
    
    # Check if close to an integer
    if math.isclose(n, round(n), rel_tol=rel_tol, abs_tol=abs_tol):
        return int(round(n))

    # Check if close to a rational approximation
    approx = bestApproximation(n, 10000)
    approx_value = approx[0]
    if approx_value is not None and math.isclose(n, approx_value, rel_tol=rel_tol, abs_tol=abs_tol):
        _, numerator, denominator, _ = approx
        return numerator / denominator

    # Check for natural constants
    #if math.isclose(n, math.pi, rel_tol=rel_tol, abs_tol=abs_tol):
    #    return math.pi
    #elif math.isclose(n, math.e, rel_tol=rel_tol, abs_tol=abs_tol):
    #    return math.e
    #elif math.isclose(n, math.sqrt(2), rel_tol=rel_tol, abs_tol=abs_tol):
    #    return math.sqrt(2)
    
    # Otherwise, round the number to a fixed precision
    return round(n, precision)


# The following are snap functions for finding a best approximated integer or rational number for a real number:
def bestApproximation(x,imax):
    """
    AI FEYNMAN based description length computations. 
    """
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x,nmax):
        x = float(x)
        c = [np.floor(x)];
        y = x - np.floor(x)
        k = 0
        while np.abs(y)!=0 and k<nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c
    
    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq`
            into a fraction, num / den
            '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num*u, num
        return num, den
    
    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))
    
    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:,0] / float(q[:,1])
    
    def truncateContFrac(q,imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k,0]), q[k,1]) <= imax:
            k = k + 1
        return q[:k]
    
    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)
    
    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x),20)),imax)
    
    if len(q) > 0:
        p = np.abs(q[:,0] / q[:,1] - abs(x)).astype(float) * (1 + np.abs(q[:,0])) * q[:,1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i,0] / float(q[i,1]), xsign* q[i,0], q[i,1], p[i])
    else:
        return (None, 0, 0, 1)

def integerSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p - np.round(p.astype(np.double)))
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.round(p.astype(np.double))[chosen])))


def zeroSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p)
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.zeros(len(chosen)))))


def rationalSnap(p, top=1):
    """Snap to nearest rational number using continued fraction."""
    p = np.array(p)
    snaps = np.array(list(bestApproximation(x,10) for x in p))
    chosen = np.argsort(snaps[:, 3])[:top]    
    d = dict(list(zip(chosen, snaps[chosen, 1:3])))
    d = {k:  f"{val[0]}/{val[1]}" for k,val in d.items()}
    
    return d

def sample_uniform(support, num_samples, rng): 
    a = np.asarray([s[0] for s in support], dtype=float)
    b = np.asarray([s[1] for s in support], dtype=float)
    return rng.uniform(low=a, high=b, size=(num_samples, a.shape[0]))

def sample_truncated_normal_vectorized(support, num_samples, rng, sigma_factor=6.0):
    a = np.asarray([s[0] for s in support], dtype=float)
    b = np.asarray([s[1] for s in support], dtype=float)
    mu, sigma = 0.5 * (a + b), (b - a) / float(sigma_factor)
    is_degenerate = sigma <= 0
    X = np.empty((num_samples, len(support)))
    if np.any(is_degenerate):
        X[:, is_degenerate] = mu[is_degenerate]
    if np.any(~is_degenerate):
        a_v, b_v, mu_v, sigma_v = a[~is_degenerate], b[~is_degenerate], mu[~is_degenerate], sigma[~is_degenerate]
        z_low, z_high = (a_v - mu_v) / sigma_v, (b_v - mu_v) / sigma_v
        X[:, ~is_degenerate] = truncnorm.rvs(z_low, z_high, loc=mu_v, scale=sigma_v,
                                            size=(num_samples, len(mu_v)), random_state=rng)
    return X

def sample_diverse(a: float, b: float, N: int, dtype: str, rng, max_parts: int = 5) -> np.ndarray: 
    assert b >= a, f'Upper interval bound {b} must be >= lower interval bound {a}.'
    if N == 0: return np.array([])
    n_parts = int(rng.integers(1, max_parts + 1))
    weights = rng.random(n_parts)
    weights /= weights.sum()
    n_points = (weights * N).astype(int)
    n_points[0] = N - n_points[1:].sum() if n_parts > 1 else N
    def sample_uniform_local(p, n): return rng.random(n) * (p[1] - p[0]) + p[0]
    def sample_normal_local(p, n): return np.full(n, p[0]) if p[1] <= 0.0 else rng.normal(loc=p[0], scale=p[1], size=n)
    def sample_uniform_int_local(p, n): return rng.integers(p[0], p[1] + 1, size=n) if p[1] >= p[0] else np.full(n, p[0], dtype=int)
    def sample_normal_int_local(p, n): return (rng.binomial(p[1] - p[0], p[2], size=n) + p[0]) if p[1] > p[0] else np.full(n, p[0], dtype=int)
    distr_dict = {'uniform': sample_uniform_local, 'normal': sample_normal_local, 'uniform_int': sample_uniform_int_local, 'normal_int': sample_normal_int_local}
    parts = []
    for i in range(n_parts):
        choices = [k for k in distr_dict.keys() if ('int' in k if dtype == 'int' else 'int' not in k)]
        distr_type = rng.choice(choices)
        n = int(n_points[i])
        if n <= 0: continue
        if distr_type == 'uniform': p = np.sort(rng.random(2)) * (b - a) + a
        elif distr_type == 'uniform_int': p = np.sort(rng.integers(int(a), int(b) + 1, size=2))
        elif distr_type == 'normal': p_mean, p_std = rng.random() * (b - a) + a, min(b - (rng.random()*(b-a)+a), (rng.random()*(b-a)+a) - a) / 3.0; p = np.array([p_mean, max(p_std, 0.0)])
        else: tmp = np.sort(rng.integers(int(a), int(b) + 1, size=2)); p = np.array([int(tmp[0]), int(tmp[1]), 0.5])
        parts.append(distr_dict[distr_type](p, n))
    x = np.concatenate(parts) if parts else np.array([])
    return np.clip(x, a, b).astype(int if dtype == 'int' else float)

def add_noise(X: np.ndarray, y: np.ndarray, noise_level: float, rng, both: bool = False) -> tuple: 
    if noise_level <= 0: return X, y
    y_noisy, X_noisy = np.copy(y), np.copy(X)
    if both:
        P = np.column_stack([X, y])
        magnitude = np.sqrt(np.mean(P**2, axis=0))
        magnitude[magnitude == 0] = 1.0
        P_noisy = P + rng.normal(0.0, noise_level * magnitude, size=P.shape)
        return P_noisy[:, :-1], P_noisy[:, -1]
    else:
        magnitude = np.sqrt(np.mean(y**2))
        if np.isfinite(magnitude) and magnitude > 0: y_noisy += rng.normal(0.0, noise_level * magnitude, size=y.shape)
        return X_noisy, y_noisy

def sample_points(expr, support, distribution, num_samples, noise_level, seed=None, rng=None, dtype='float'): 
    if rng is None: rng = np.random.default_rng(seed)
    if distribution == "uniform": X = sample_uniform(support, num_samples, rng)
    elif distribution == "normal": X = sample_truncated_normal_vectorized(support, num_samples, rng)
    elif distribution == "diverse": X = np.column_stack([sample_diverse(a=a, b=b, N=num_samples, dtype=dtype, rng=rng) for a, b in support])
    else: raise ValueError(f"Distribution '{distribution}' not supported.")
    y = predict(expr, X) 
    if noise_level > 0: X, y = add_noise(X, y, noise_level, rng)
    return X, y

# --- Step 1: Robust Sampling Wrapper ---
def sample_valid_points(expr, support, distribution, num_samples, noise_level, rng,
                        dtype='float', sampling_factor=1.5, max_tries=100):
    if num_samples == 0:
        return np.empty((0, len(support))), np.empty((0,))
    collected_X, collected_y = [], []
    for _ in range(max_tries):
        n_collected = len(collected_y[0]) if collected_y else 0
        n_needed = num_samples - n_collected
        if n_needed <= 0: break
        n_to_sample = max(10, int(n_needed * sampling_factor))
        X_batch, y_batch = sample_points(expr, support, distribution, n_to_sample, 
                                         noise_level, rng=rng, dtype=dtype)
        valid_mask = np.isfinite(y_batch)
        if np.any(valid_mask):
            collected_X.append(X_batch[valid_mask])
            collected_y.append(y_batch[valid_mask])
    if not collected_y: raise RuntimeError(f"Failed to generate any valid points after {max_tries} tries.")
    final_X, final_y = np.vstack(collected_X), np.concatenate(collected_y)
    if final_y.shape[0] < num_samples: raise RuntimeError(f"Failed to generate {num_samples} valid points. Collected {final_y.shape[0]}.")
    return final_X[:num_samples], final_y[:num_samples]

# --- Step 2: Domain Exclusion Sampler ---
def _get_hyperbox_volume(support):
    return np.prod([max(0, s[1] - s[0]) for s in support])

def sample_points_excluding_box(expr, outer_support, inner_support, distribution, 
                                num_samples, noise_level, rng, dtype='float', max_tries=100):
    if num_samples == 0:
        return np.empty((0, len(outer_support))), np.empty((0,))
    inner_low = np.asarray([s[0] for s in inner_support], dtype=float)
    inner_high = np.asarray([s[1] for s in inner_support], dtype=float)
    vol_outer, vol_inner = _get_hyperbox_volume(outer_support), _get_hyperbox_volume(inner_support)
    if vol_outer <= 0: raise ValueError("Outer support volume is zero.")
    acceptance_rate = 1.0 - (vol_inner / vol_outer)
    if acceptance_rate <= 1e-9: raise ValueError("Inner support covers the entire outer support.")
    sampling_factor = 1.2 / acceptance_rate
    collected_X, collected_y = [], []
    for _ in range(max_tries):
        n_collected = len(collected_y[0]) if collected_y else 0
        n_needed = num_samples - n_collected
        if n_needed <= 0: break
        n_to_propose = max(10, int(n_needed * sampling_factor))
        X_batch, y_batch = sample_valid_points(expr=expr, support=outer_support, distribution=distribution, 
                                               num_samples=n_to_propose, noise_level=noise_level, 
                                               rng=rng, dtype=dtype)
        is_outside_mask = ~np.all((X_batch >= inner_low) & (X_batch <= inner_high), axis=1)
        if np.any(is_outside_mask):
            collected_X.append(X_batch[is_outside_mask])
            collected_y.append(y_batch[is_outside_mask])
    if not collected_y: raise RuntimeError(f"Failed to generate any OOD points after {max_tries} tries.")
    final_X, final_y = np.vstack(collected_X), np.concatenate(collected_y)
    if final_y.shape[0] < num_samples: raise RuntimeError(f"Failed to generate {num_samples} OOD points. Collected {final_y.shape[0]}.")
    return final_X[:num_samples], final_y[:num_samples]

# --- Step 3: Sub-box Sampler ---
def sample_sub_box_within_domain(enclosing_support, rng, min_width_factor=0.25):
    sub_box_support = []
    for (a, b) in enclosing_support:
        parent_width = b - a
        if parent_width <= 0:
            sub_box_support.append((a, b))
            continue
        min_required_width = parent_width * min_width_factor
        while True:
            p1, p2 = rng.uniform(low=a, high=b, size=2)
            new_min, new_max = min(p1, p2), max(p1, p2)
            if new_max - new_min >= min_required_width:
                sub_box_support.append((new_min, new_max))
                break
    return sub_box_support

def sample_disjoint_ood_boxes(total_support, excluded_support, rng):
    """
    Creates two disjoint hyperboxes for OOD training and testing. (Robust Version)

    This version handles cases where the excluded domain is flush with the edge
    of the total domain by partitioning the single available region if necessary.
    """
    num_dimensions = len(total_support)
    split_dim = rng.integers(num_dimensions)

    total_min, total_max = total_support[split_dim]
    excluded_min, excluded_max = excluded_support[split_dim]

    left_slab = (total_min, excluded_min)
    right_slab = (excluded_max, total_max)

    left_width = left_slab[1] - left_slab[0]
    right_width = right_slab[1] - right_slab[0]

    train_region = list(total_support)
    ood_region = list(total_support)

    if left_width > 0 and right_width > 0:
        # Case A: Space on both sides. Randomly assign one slab to train, one to ood.
        if rng.random() > 0.5:
            train_region[split_dim], ood_region[split_dim] = left_slab, right_slab
        else:
            train_region[split_dim], ood_region[split_dim] = right_slab, left_slab
            
    elif left_width > 0: # Only space on the left
        # Case B: Partition the single available left slab.
        split_point = rng.uniform(left_slab[0], left_slab[1])
        sub_slab1 = (left_slab[0], split_point)
        sub_slab2 = (split_point, left_slab[1])
        
        if rng.random() > 0.5:
            train_region[split_dim], ood_region[split_dim] = sub_slab1, sub_slab2
        else:
            train_region[split_dim], ood_region[split_dim] = sub_slab2, sub_slab1

    elif right_width > 0: # Only space on the right
        # Case B: Partition the single available right slab.
        split_point = rng.uniform(right_slab[0], right_slab[1])
        sub_slab1 = (right_slab[0], split_point)
        sub_slab2 = (split_point, right_slab[1])

        if rng.random() > 0.5:
            train_region[split_dim], ood_region[split_dim] = sub_slab1, sub_slab2
        else:
            train_region[split_dim], ood_region[split_dim] = sub_slab2, sub_slab1

    else:
        # Case C: No space anywhere. This is a fatal configuration.
        raise ValueError(
            f"Cannot create any OOD boxes. The excluded domain {excluded_support} "
            f"covers the entire total domain {total_support}."
        )

    # Finally, sample sub-boxes from within the determined, disjoint regions.
    train_box_support = sample_sub_box_within_domain(train_region, rng)
    ood_test_box_support = sample_sub_box_within_domain(ood_region, rng)
    
    return train_box_support, ood_test_box_support

def is_box_plausible(expr, support, rng, num_check_points=100, min_success_rate=0.01):
    """
    Quickly checks if a given support box is likely to produce valid points.
    """
    if _get_hyperbox_volume(support) <= 0:
        return False
    # Use the basic, non-robust sample_points for a quick check
    try:
        _, y_check = sample_points(expr, support, "uniform", num_check_points, 0, rng=rng)
        success_rate = np.mean(np.isfinite(y_check))
        return success_rate >= min_success_rate
    except Exception:
        # If even the check fails for some reason, the box is not plausible.
        return False

def sample_sub_box_within_domain_validated(expr, enclosing_support, rng, max_tries=1000, **kwargs):
    """A version of sample_sub_box that validates the output."""
    for _ in range(max_tries):
        candidate_box = sample_sub_box_within_domain(enclosing_support, rng, **kwargs)
        if is_box_plausible(expr, candidate_box, rng):
            return candidate_box
    raise RuntimeError(f"Failed to find a plausible sub-box after {max_tries} attempts.")

def sample_disjoint_ood_boxes_validated(expr, total_support, excluded_support, rng, max_tries=1000):
    """A version of sample_disjoint_ood_boxes that validates both output boxes."""
    for _ in range(max_tries):
        # This function can fail with a ValueError if geometry is impossible
        try:
            cand_train, cand_ood = sample_disjoint_ood_boxes(total_support, excluded_support, rng)
            if is_box_plausible(expr, cand_train, rng) and is_box_plausible(expr, cand_ood, rng):
                return cand_train, cand_ood
        except ValueError:
            # Pass on this attempt if the initial geometry was invalid
            continue
    raise RuntimeError(f"Failed to find two plausible disjoint OOD boxes after {max_tries} attempts.")

def generate_problem(task, domain_box, sampling_distribution, num_train_points, 
                     train_domain_bounds=(-10, 10), total_domain_bounds=(-100, 100), 
                     seed=42, test_set_ratio=0.25, noise_level=0):
    try:
        expr_true = sympy.parse_expr(str(task["expression"]))
    except:
        expr_true = parse_expr(str(task["expression"]), transformations=(standard_transformations + (convert_xor,)))

    num_dimensions = len(task["support"])
    num_test_points = int(num_train_points * test_set_ratio)
    
    rng = np.random.default_rng(seed)
    
    pre_training_domain_support = [train_domain_bounds for _ in range(num_dimensions)]
    total_domain_support = [total_domain_bounds for _ in range(num_dimensions)]

    # --- Cases for domain_box ---
    if domain_box == "in_domain":
        train_support = pre_training_domain_support
        
        total_id_points = num_train_points + num_test_points
        X_id_total, y_id_total = sample_valid_points(
            expr_true, train_support, sampling_distribution, total_id_points, noise_level, rng)
        X_train, y_train = X_id_total[:num_train_points], y_id_total[:num_train_points]
        X_test_id, y_test_id = X_id_total[num_train_points:], y_id_total[num_train_points:]
        
        X_test_ood, y_test_ood = sample_points_excluding_box(
            expr_true, total_domain_support, train_support, sampling_distribution, 
            num_test_points, noise_level, rng)

    elif domain_box == "within_domain":
        train_support = sample_sub_box_within_domain_validated(expr_true, pre_training_domain_support, rng)

        total_id_points = num_train_points + num_test_points
        X_id_total, y_id_total = sample_valid_points(
            expr_true, train_support, sampling_distribution, total_id_points, noise_level, rng)
        X_train, y_train = X_id_total[:num_train_points], y_id_total[:num_train_points]
        X_test_id, y_test_id = X_id_total[num_train_points:], y_id_total[num_train_points:]
        
        X_test_ood, y_test_ood = sample_points_excluding_box(
            expr_true, total_domain_support, train_support, sampling_distribution, 
            num_test_points, noise_level, rng)

    elif domain_box == "out_of_domain":
        # Sample two disjoint boxes, both outside the pre-training domain.
        train_support, ood_support = sample_disjoint_ood_boxes_validated(
            expr_true, total_domain_support, pre_training_domain_support, rng)
        
        # Training and ID test data are sampled from the first OOD box
        total_train_points = num_train_points + num_test_points
        X_id_total, y_id_total = sample_valid_points(
            expr_true, train_support, sampling_distribution, total_train_points, noise_level, rng)
        
        X_train, y_train = X_id_total[:num_train_points], y_id_total[:num_train_points]
        X_test_id, y_test_id = X_id_total[num_train_points:], y_id_total[num_train_points:]
        
        # The OOD test set is sampled from the *second*, disjoint OOD box
        X_test_ood, y_test_ood = sample_valid_points(
            expr_true, ood_support, sampling_distribution, num_test_points, noise_level, rng)
        
    else:
        raise ValueError(f"domain_box='{domain_box}' not recognized.")
        
    return expr_true, X_train, y_train, X_test_id, y_test_id, X_test_ood, y_test_ood


#####################################
# Symbolic Checks
#####################################


def tree_size(expr) -> int:
    '''
    Counts number of nodes in expression tree.

    @Params:
        expr... sympy expression

    @Returns:
        number of nodes in expression tree
    '''

    children = expr.args
    return 1 + sum([tree_size(c) for c in children])

def is_const(expr, timeout:float = 5.0) -> bool:
    '''
    Checks whether sympy expression is a constant.

    @Params:
        expr... sympy expression
        timeout... number of seconds after which to abort

    @Returns:
        True, if expression could successfully be reduced to a constant
    '''

    ret = False
    if tree_size(expr) > 50:
        return (len(expr.free_symbols) == 0)
    try:
        with stopit.ThreadingTimeout(timeout, swallow_exc=False) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            ret = expr.is_constant()
        if to_ctx_mgr:
            return ret
        else:
            return (len(expr.free_symbols) == 0)
    except (TypeError, KeyError, AttributeError, RecursionError, stopit.utils.TimeoutException, stopit.TimeoutException):
        return (len(expr.free_symbols) == 0)  
    

def round_floats(ex1, round_digits:int = 3, max_v:int = np.inf):
    '''
    Rounds floats within sympy expression.

    @Params:
        ex1... sympy expression
        max_v... numbers greater are set to infinity
    
    @Returns:
        sympy expression
    '''

    ex2 = ex1.evalf()
    found = True
    max_rounds = 3
    n_rounds = 0

    while found and n_rounds < max_rounds:
        n_rounds += 1
        found = False
        try:
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Number) or isinstance(a, sympy.NumberSymbol):
                    if abs(a) > max_v:
                        if a > 0:
                            ex2 = ex2.subs(a, sympy.oo)
                        else:
                            ex2 = ex2.subs(a, -sympy.oo)
                        ex2 = sympy.cancel(ex2)
                    elif abs(round(a) - float(a)) <= 1/(10**round_digits):
                        found = True
                        ex2 = ex2.subs(a, sympy.Integer(round(a)))
                    else:
                        ex2 = ex2.subs(a, sympy.Float(round(a, round_digits), round_digits))
        except (TypeError, ValueError):
            found = False
        ex1 = ex2
    return ex1

def get_subexprs_sympy(expr) -> list:
    '''
    Splits an expression into subexpressions.
    @Params:
        expr... sympy expression
    @Returns:
        list of all subexpressions according to sympy
    '''

    func = expr.func
    children = expr.args
    subs = [expr]
    for c in children:
        subs += get_subexprs_sympy(c)
    return subs


def is_substitution(expr, subexpr) -> bool:
    '''
    Given an expression and a subexpression, checks if the subexpression is a valid substitution.
    Example:
    For a given expression x_0*x_1/x_2 the following subexpressions are valid substitutions:
    - x_0*x_1 -> z/x_2
    - x_0/x_2 -> z*x_1
    - x_1/x_2 -> x_0*z

    
    @Params:
        expr... sympy expression
        subexpr... sympy expression

    @Returns:
        True, if subexpr is a valid substitution
    '''

    # assumption: variables are named x_i (no curly brackets)
    subs_idx = sorted([int(str(x).split('_')[-1]) for x in subexpr.free_symbols if '_' in str(x)])
    z = sympy.Symbol('z')
    repl_expr = expr.subs({
        subexpr: z, # ident
        1/subexpr: 1/z, # inverse
        -subexpr: -z, # negate
        subexpr**2 : z**2, # square
        sympy.sqrt(subexpr) : sympy.sqrt(z), # square-root
        sympy.sin(subexpr) : sympy.sin(z), # sine
        sympy.cos(subexpr) : sympy.cos(z), # cosine
        sympy.log(subexpr) : sympy.log(z), # log
        sympy.exp(subexpr) : sympy.exp(z), # exp
    } )

    for x in repl_expr.free_symbols:
        if '_' in str(x):
            if int(str(x).split('_')[-1]) in subs_idx:
                return False
    return True

def eval_expr(expr, X):
    x_symbs = [f'x_{i}' for i in range(X.shape[1])]
    exec_func = sympy.lambdify(x_symbs, expr)
    pred = exec_func(*[X[:, i] for i in range(X.shape[1])])
    if isinstance(pred, numbers.Number):
        pred = pred*np.ones(X.shape[0])
    return pred

#####################################
# Sympy -> Expression Trees
#####################################

class Node:
    def __init__(self, arg, left = None, right = None, done = False):
        self.arg = arg
        self.left = left
        self.right = right
        self.done = done

class Tree:
    def __init__(self, root):
        self.root = root

def preorder(node, tokens = []):
    tokens.append(node.arg)
    if node.left is not None:
        tokens = preorder(node.left, tokens)
    if node.right is not None:
        tokens = preorder(node.right, tokens)
    return tokens
    
def node2string(node):
    if node.left is None and node.right is None:
        return str(node.arg)
    else:
        if node.right is None:
            # unary 
            op = f'{node.arg}({node2string(node.left)})'
            op = op.replace('inv', '1/')
            op = op.replace('neg', '-')
        else:
            # binary
            op = f'({node2string(node.left)}){node.arg}({node2string(node.right)})'
        return op

def get_leafes(node, dir_str = '', leaf_nodes = []):
    if node.left is None and node.right is None:
        leaf_nodes.append((node, dir_str))
    else:
        if node.left is not None:
            leaf_nodes = get_leafes(node.left, dir_str + 'l', leaf_nodes)
        if node.right is not None:
            leaf_nodes = get_leafes(node.right, dir_str + 'r', leaf_nodes)
    
    return leaf_nodes

def follow_path(tree, dir_str):
    node = tree.root
    for s in dir_str:
        if s == 'l':
            node = node.left
        else:
            node = node.right
    return node

def set_node(tree, dir_str, node):
    if dir_str == '':
        tree.root = node
    else:
        # follow except for last
        parent_node = tree.root
        for s in dir_str[:-1]:
            if s == 'l':
                parent_node = parent_node.left
            else:
                parent_node = parent_node.right
        if dir_str[-1] == 'l':
            parent_node.left = node
        else:
            parent_node.right = node
    return tree

def get_variants(expr):
    variants = []
    func, args = expr.func, expr.args
    # transformations for power
    if func is sympy.core.power.Pow:
        if ('x_' in str(args[1])) or (int(args[1]) != float(args[1]) and args[1] != 0.5 and args[1] > 0):
            func = sympy.exp
            if ('x_' in str(args[1])):
                args = [args[1]*sympy.log(args[0])]
            else:
                args = [float(args[1])*sympy.log(args[0])]
            
        elif int(args[1]) == float(args[1]) and args[1] > 0:
            if args[1] < 10:
                func = sympy.core.mul.Mul
                args = [args[0]]*int(args[1])
            else:
                func = sympy.exp
                args = [int(args[1])*sympy.log(args[0])]
    
            
    if (func == sympy.core.mul.Mul) and args[0] == -1:
        # negation
        symb = 'neg'
        child_expr = 1
        for i in range(1, len(args)):
            child_expr = child_expr * args[i]
        child = Node(child_expr)
        variants.append(Node(symb, child, done = True))
    elif (func == sympy.core.mul.Mul):
        # multiplication
        symb = '*'
        if (len(args) > 2):
            k = len(args)
            for i in range(1, 2**k//2):
                s = format(i, f'0{k}b')
                left = 1
                right = 1
                for j in range(k):
                    if s[j] == '1':
                        left *= args[j]
                    else:
                        right *= args[j]
                variants.append(Node(symb, Node(left), Node(right), done = True))
        else:
            left = Node(args[0])
            right = Node(args[1])
            variants.append(Node(symb, left, right, done = True))
            
    elif (func == sympy.core.add.Add):
        # addition
        symb = '+'
        if (len(args) > 2):
            k = len(args)
            for i in range(1, 2**k-1):
                s = format(i, f'0{k}b')
                left = 0
                right = 0
                for j in range(k):
                    if s[j] == '1':
                        left += args[j]
                    else:
                        right += args[j]
                variants.append(Node(symb, Node(left), Node(right), done = True))
        else:
            left = Node(args[0])
            right = Node(args[1])
            variants.append(Node(symb, left, right, done = True))
    elif (func == sympy.core.power.Pow) and (args[1] < 0):
        # inversion
        symb = 'inv'
        child = Node(args[0]**(-args[1]))
        variants.append(Node(symb, child, done = True))
    elif (func == sympy.core.power.Pow) and (args[1] == 0.5):
        # sqrt
        symb = 'sqrt'
        child = Node(args[0])
        variants.append(Node(symb, child, done = True))
    elif (func == sympy.core.power.Pow) and (args[1] == 0.5):
        # sqrt
        symb = 'sqrt'
        child = Node(args[0])
        variants.append(Node(symb, child, done = True))
    elif len(args) == 0:
        # leaf
        symb = str(expr)
        variants.append(Node(symb, done = True))
    else:
        # other unary operations
        assert len(args) == 1, f'{func}, {args}'
        symb = str(func)
        child = Node(args[0])
        variants.append(Node(symb, child, done = True))
    return variants

def develop(tree, dir_str):
    # find leaf node
    node = follow_path(tree, dir_str)
    expr = node.arg
    assert node.left is None and node.right is None

    # create variants
    variants = get_variants(expr)
    
    # create trees
    variant_trees = []
    for variant_node in variants:
        variant_tree = copy.deepcopy(tree)
        variant_tree = set_node(variant_tree, dir_str, variant_node)
        variant_trees.append(variant_tree)

    return variant_trees

def create_trees(expr, max_trees = 1000):
    node = Node(expr, done = False)
    tree = Tree(node)
    open_tree_list = [tree]
    finished_trees = []

    running = True
    while len(open_tree_list) > 0 and running:
        tree = open_tree_list.pop()
        leafes = get_leafes(tree.root, dir_str = '', leaf_nodes = [])
        finished = True
        for node, dir_str in leafes:
            if not node.done:        
                trees = develop(tree, dir_str)
                open_tree_list += trees
                finished = False
                break
        if finished:
            finished_trees.append(copy.deepcopy(tree))
            if len(finished_trees) == max_trees:
                running = False
    return finished_trees
    


from timeit import default_timer as timer
import multiprocessing
from typing import Iterable, Tuple
import numbers
import itertools
import math 
import warnings

import numpy as np
import sympy as sp
from scipy.stats import truncnorm

from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

try:
    import regressors.zss as zss
    import regressors.metrics as metrics
except: 
    import zss
    import metrics

# ignore warnings
warnings.filterwarnings("ignore")

def queued_func(func, queue: multiprocessing.Queue, args=(), kwargs=None):
    kwargs = {} if kwargs is None else dict(kwargs)
    try:
        # Try the normal call first
        try:
            result = func(*args, **kwargs)
        except TypeError:
            # Fallbacks if the function didn't accept our kwargs:
            # prefer common names, then positional from kwargs values
            if 'expr' in kwargs:
                result = func(kwargs['expr'])
            elif 'e' in kwargs:
                result = func(kwargs['e'])
            else:
                # last resort: call with positional values of kwargs
                result = func(*tuple(kwargs.values()))
    except Exception as exc:
        # Put the exception in the queue so the parent can inspect it
        queue.put({"result": None, "error": exc})
    else:
        queue.put({"result": result})

def predict(expr:sympy.Expr, X:np.ndarray) -> np.ndarray:
    """
    Evaluates expression at datapoints X.

    Params:
        expr... sympy expression
        X... sample matrix (n_samples x n_dim)
    
    Returns:
        predictions as numpy array (n_samples)
    """
    ret = None
    try:
        ret = eval_expr_numpy(expr, X)
    except Exception as e:
        pass

    # if failed, try to convert to sympy first
    if type(expr) == str:
        expr = sympy.sympify(expr)
    try:
        ret = eval_expr_numpy(expr, X)
    except Exception as e:
        pass

    # last resort: sympy eval
    try:
        ret = eval_expr(expr, X)
    except Exception as e:
        pass
    if isinstance(ret, numbers.Number):
        ret = ret*np.ones(X.shape[0])
    return ret

def expr2numpy(expr:sympy.Expr, d:int) -> str:
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
        'sin' : 'np.sin',
        'cos' : 'np.cos',
        'log' : 'np.log',
        'ln' : 'np.log',
        'asin' : 'np.arcsin',
        'tan' : 'np.tan',
        'cot' : '1/np.tan',
        'atan' : 'np.arctan',
        'sqrt' : 'np.sqrt',
        'pi' : 'np.pi',
        'E' : 'np.e',
        'Abs' : 'np.abs',
    }
    expr_str = str(expr)
    for s in repl_dict:
        expr_str = expr_str.replace(s, repl_dict[s])
    for i in range(d):
        expr_str = expr_str.replace(f'x_{i}', f'X[:, {i}]')
    return expr_str

def eval_expr_numpy(expr:sympy.Expr, X:np.ndarray) -> np.ndarray:
    """
    Evaluate an expression using numpy (fast).

    Params:
        expr... expression
        X... data on which to evaluate
    
    Returns:
        y = expr(X) as numpy array
    """
    np_str = expr2numpy(str(expr), X.shape[1])
    y = eval(np_str)
    return y

def make_poly_models(max_degree=3, standardize=True):
    """
    Return a list of (name, model) pairs for degrees 1..max_degree inclusive.
    degree=1 will be simple linear (implemented via PolynomialFeatures(degree=1) for uniformity).
    """
    models = []
    for deg in range(1, max_degree + 1):
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        # include_bias=False so degree=1 behaves like linear w/out constant duplication
        steps.append(("poly", PolynomialFeatures(degree=deg, include_bias=False)))
        steps.append(("ridge", LinearRegression()))
        pipeline = Pipeline(steps)
        name = f"poly_{deg}"
        models.append(pipeline)
    return models

def train_test_split_NN(X:np.ndarray, y:np.ndarray, test_size:float = 0.2, k_neighbors:int = 5) -> tuple:
    # für jeden punkt, messe den mean abstand zu den k nächsten nachbarn (k=5)
    # nimm dann die punkte, die die weitesten abstände zu ihren k nächsten nachbarn haben (=die am abgelegensten sind). 
    testsize = int(test_size*X.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # nearest neighbor distance (excluding self, so take column 1)
    nn_distances = np.mean(distances[:, 1:], axis = 1)

    # Select indices of the largest distances
    test_idx = np.argsort(-nn_distances)[:testsize]

    test_mask = np.zeros(len(y)).astype(bool)
    test_mask[test_idx] = True

    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

def train_test_split_gp(X:np.ndarray, y:np.ndarray, test_size:float = 0.2) -> tuple:
    """
    Fit a GP with RBF kernel and find the region with highest predictive variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features.
    y : ndarray of shape (n_samples,)
        Targets.
    test_size : float
        Size of test region to select.

    Returns
    -------
    best_test_idx : ndarray
        Indices of points for the test region.
    """

    k_neighbors = int(test_size*len(X))

    # Fit GP
    gp = GaussianProcessRegressor(normalize_y=True)
    gp.fit(X, y)

    # Compute posterior variance (std) at each sample point
    _, std = gp.predict(X, return_std=True)
    var = std ** 2  # variance at each point

    # Aggregate variance over neighborhoods
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    _, neigh_idx = nbrs.kneighbors(X)

    agg_var = []
    for idx_list in neigh_idx:
        agg_var.append(var[idx_list].mean())  # mean variance in neighborhood
    agg_var = np.array(agg_var)

    # Pick region with highest aggregated variance
    center_idx = np.argmax(agg_var)
    best_test_idx = neigh_idx[center_idx]

    test_mask = np.zeros(len(y)).astype(bool)
    test_mask[best_test_idx] = True

    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

def train_test_split_r2(X:np.ndarray, y:np.ndarray, test_size:float = 0.2, models:list = [], random_state:int = None, max_it:int = 10) -> tuple:
    """
    For each candidate point, define testset as its nearest neighbors, fit models on train split,
    compute mean R^2 on test split; pick split with lowest maximum R^2.

    Parameters
    ----------
    models : list of sklearn-like regressors (unfitted or templates that can be cloned)
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    test_size : float
        share of testdata
    models : list
        List of sklearn-like regressors (unfitted or templates that can be cloned).
    random_state : int or None
        Random seed for reproducibility.

    Usage: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if len(models) == 0:
        models = make_poly_models()
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    testsize = int(test_size*n)
    if testsize >= n:
        raise ValueError("testsize must be less than n_samples")

    nbrs = NearestNeighbors(n_neighbors=testsize).fit(X)

    best_score = np.inf
    best_test_idx = None

    idxs = np.arange(n)
    np.random.shuffle(idxs)
    idxs = idxs[:max_it]

    for idx in idxs:
        _, neigh_idx = nbrs.kneighbors(X[[idx]])
        test_idx = neigh_idx.ravel()
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train and evaluate each model
        r2s = []
        for m in models:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r2s.append(r2_score(y_test, y_pred))

        agg_r2 = np.max(r2s)
        if agg_r2 < best_score:
            best_score = agg_r2
            best_test_idx = test_idx

    best_train_idx = np.setdiff1d(np.arange(n), best_test_idx, assume_unique=True)
    return X[best_train_idx], X[best_test_idx], y[best_train_idx], y[best_test_idx]

def train_test_split_mean(X:np.ndarray, y:np.ndarray, test_size:float = 0.2) -> tuple:
    # take points with highest distance to mean as extrapolation points
    train_amount = int((1-test_size)*len(X))
    sort_idxs = np.argsort(np.linalg.norm(X - np.mean(X, axis = 0), axis = 1))
    train_idxs = sort_idxs[:train_amount]
    test_idxs = sort_idxs[train_amount:]
    X_train, y_train = X[train_idxs], y[train_idxs]
    X_test, y_test = X[test_idxs], y[test_idxs]
    return X_train, X_test, y_train, y_test


def get_train_test_split(X, y, test_size, split_method="mean", seed=None):
    """
    """
    
    # 1. get the id, ood test split 
    if split_method == "mean":
        X_pool, X_test_ood, y_pool, y_test_ood = train_test_split_mean(X, y, test_size)
    elif split_method == "nn":
        X_pool, X_test_ood, y_pool, y_test_ood = train_test_split_nn(X, y, test_size)
    elif split_method == "gp":
        X_pool, X_test_ood, y_pool, y_test_ood = train_test_split_gp(X, y, test_size)
    elif split_method == "r2":
        X_pool, X_test_ood, y_pool, y_test_ood = train_test_split_r2(X, y, test_size)
    else:
        raise ValueError(f"Unknown split method: {split_method}")

    # 2. get the id train test split 
    X_train, X_test_id, y_train, y_test_id = train_test_split(
        X_pool, y_pool, test_size=test_size, random_state=seed
    )

    return X_train, X_test_id, X_test_ood, y_train, y_test_id, y_test_ood

def replace_numbers_in_expr_with_placeholders(expr:sympy.Expr) -> sympy.Expr: 
    """
    Replaces all numbers in the expression with placeholders (c_1, c_2, ...).
    
    Parameters:
        expr: sympy expression
    
    Returns:
        expression with numbers replaced by placeholders
    """
    expr = sympy.sympify(expr)  # Ensure the input is a sympy expression
    
    # Function to identify atomic numbers
    is_atomic_number = lambda e: e.is_Atom and e.is_number
    
    # Collect all unique numbers in the expression
    unique_numbers = list({subexpr for subexpr in sympy.preorder_traversal(expr) if is_atomic_number(subexpr)})
    
    # Create a mapping from numbers to placeholders
    mapping = {f"c_{i+1}": num for i, num in enumerate(unique_numbers)}
    
    # Create substitution rules
    substitution = {num: sympy.Symbol(placeholder) for placeholder, num in mapping.items()}
    
    # Perform the substitution
    modified_expr = expr.subs(substitution)
    
    return modified_expr

def get_alternatives(expr:sympy.Expr, timeout:float = 0.1) -> list:
    '''
    Returns alternative formulations of the same expression.
    Params:
        expr... sympy expression

    Returns:
        list of different formulations according to sympy
    '''
    variant_funcs = [sympy.expand, sympy.factor, sympy.cancel, sympy.apart, 
                  sympy.trigsimp, sympy.expand_trig,
                  sympy.powsimp, sympy.expand_power_exp, sympy.expand_power_base, sympy.powdenest,
                  sympy.expand_log, sympy.logcombine]
    seen = [str(expr)]
    res = [expr]
    for func in variant_funcs:

        # timed out check
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=queued_func, args=(func, q, {"expr": expr}))
        p.start()
        p.join(timeout=timeout/len(variant_funcs))
        p.terminate()

        if p.exitcode is None:
            # Process is still alive, indicating a timeout
            tmp = str(expr)
        else:
            # Get the result from the queue
            result = q.get(block=False)
            tmp = result["result"]
            
        tmp_str = str(tmp)
        if tmp_str not in seen:
            res.append(tmp)
            seen.append(tmp_str)
    return res

def get_subexprs_sympy(expr:sympy.Expr) -> list:
    '''
    Splits an expression into subexpressions.
    
    Params:
        expr... sympy expression

    Returns:
        list of all subexpressions according to sympy
    '''

    # check if expr is a string
    if isinstance(expr, str):
        print("Warning: expr is a string. Converting to sympy expression.")
        expr = sympy.sympify(expr)

    if not expr.args:
        return [expr]
    children = expr.args
    subs = [expr]
    for c in children:
        subs += get_subexprs_sympy(c)
    return subs

def expr2tree(expr:sympy.Expr) -> zss.Node:
    op = str(expr.func)
    children = expr.args
    if len(children) == 0:
        return zss.Node(op)
    else:
        ret = zss.Node(op)
        for child in children:
            ret.addkid(expr2tree(child))
        return ret

def count_nodes(zss_node:zss.Node) -> int:
    '''
    Counts the number of nodes in a tree.

    Params:
        zss_node... zss node

    Returns:    
        number of nodes in the tree
    '''
    if zss_node is None:
        return 0
    count = 1
    for child in zss_node.children:
        count += count_nodes(child)
    return count

def is_const_symbolic(expr:sympy.Expr) -> bool:
    '''
    Checks whether sympy expression is a constant.

    @Params:
        expr... sympy expression

    @Returns:
        True, if expression is a constant
    ''' 
    if len(expr.free_symbols) == 0:
        return True
    else:
        # make a numeric check 
        d = max([int(str(s).split('_')[-1]) for s in expr.free_symbols])+1
        X = np.random.rand(1000, d)*200 - 100
        pred = predict(expr, X)

        # If the result is None (due to NaN/inf), treat as non-constant
        if pred is None:
            return False

        # If we have more than 2 valid points, we can check for constancy
        mask = np.isfinite(pred)
        if mask.sum() > 2:
            pred = pred[mask]
            # Variance should be zero
            return np.isclose(np.var(pred), 0)
        else:
            # If we have less than 2 valid points, we cannot check for constancy
            return False

def is_constant_non_zero(expr:sp.Expr, atol:float=1e-8, rtol:float=1e-5) -> bool:
    """
    """
    if len(expr.free_symbols) == 0:
        return not expr.is_zero
    else:
        # check if the expression ratio is a constant != 0 
        #d = max([int(str(s).split('_')[-1]) for s in expr.free_symbols])+1
        available_dims = [int(str(s).split('_')[-1]) for s in expr.free_symbols if str(s).startswith('x')]
        if not available_dims:
            return True
        d = max(available_dims)+1
        X = np.random.rand(1000, d)*200 - 100 #X = np.random.rand(10000, d)*200 - 100
        pred_values = predict(expr, X)

        # If the result is None (due to NaN/inf), treat as non-constant
        if pred_values is None:
            return False

        # If we have more than 2 valid points, we can check for constancy
        finite_mask = np.isfinite(pred_values)
        if finite_mask.sum() > 2:
            pred_finite = pred_values[finite_mask]

            # Compare each element of pred to the mean of pred within the tolerance
            mean_val = np.mean(pred_finite)
            is_numerically_constant = np.all(np.isclose(pred_finite, mean_val, atol=atol, rtol=rtol))

            if is_numerically_constant:
                # check if the quotient is a constant which is zero 
                is_value_zero = np.isclose(mean_val, 0, atol=atol, rtol=rtol)

                if is_value_zero:
                    # quotient is zero -> not allowed case -> return False
                    return False
                else:
                    # quotient is a valid constant which is not zero 
                    return True
            else:
                # if quotient is not a constant, return False
                return False
        else:
            # If we have less than 2 valid points, we cannot check for constancy
            return False

def get_number_complexity(number, use_number_snapping=False, rel_tol=1e-9, abs_tol=1e-10) -> float:
    '''
    Get the complexity of a number in bits. 

    Args:
        number (float): the number to get the complexity for
        use_number_snapping (bool): if True, snap the number to the nearest integer or rational number if the difference is smaller than epsilon.

    Returns:
        bit_complexity (float): 
    '''
    n = float(number)
    
    # check nan
    if np.isnan(n):
        raise ValueError("Number is NaN")

    # Check if close to an integer
    if math.isclose(n, round(n), rel_tol=rel_tol, abs_tol=abs_tol):
        return np.log2(1 + np.float64(abs(int(round(n))))) # IMPORTANT: for some reason np.log2 only works on float64 ... so we have to turn the integer back to float64 to make it work

    # Check if close to a rational approximationy
    if use_number_snapping:
        approx = bestApproximation(n, 10000)
        approx_value = approx[0]
        if approx_value is not None and math.isclose(n, approx_value, rel_tol=rel_tol, abs_tol=abs_tol):
            _, numerator, denominator, _ = approx
            return np.log2((1 + abs(numerator)) * abs(denominator))
    
    # check natural constants
    if math.isclose(n, math.pi, rel_tol=rel_tol, abs_tol=abs_tol):
        return np.log2(1+3)

    #else:
    #    PrecisionFloorLoss = 1e-14
    #    return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2

    # otherwise: expect real number of given precision, approximate real number with given procision by scientific notation -> approximate bits roughly by magnitude of number
    # Handle negative numbers
    n = abs(n)

    # Get the base-10 exponent
    exponent = int(math.floor(math.log10(n)))

    # Calculate the mantissa
    mantissa = n / (10 ** exponent)

    # Calculate mantissa bits -> we use an approximation of the magnitude of the number here and discard the fractional precision
    mantissa_bits = np.log2(mantissa)

    # Calculate exponent bits -> we use an approximation of the magnitude here 
    exponent_bits = np.log2(math.pow(10, abs(exponent)))

    # Total bits = mantissa + exponent + sign
    return mantissa_bits + exponent_bits + 1  # Add 1 for the sign bit


################################################################
# The following code is taken from AIFeynman
# https://github.com/SJ001/AI-Feynman/blob/master/aifeynman/S_snap.py
def bestApproximation(x:float, imax:int):
    """
    AI FEYNMAN based description length computations. 
    https://github.com/SJ001/AI-Feynman/blob/master/aifeynman/S_snap.py
    """
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x,nmax):
        x = float(x)
        c = [np.floor(x)]
        y = x - np.floor(x)
        k = 0
        while np.abs(y)!=0 and k<nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c
    
    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq`
            into a fraction, num / den
            '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num*u, num
        return num, den
    
    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))
    
    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:,0] / float(q[:,1])
    
    def truncateContFrac(q,imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k,0]), q[k,1]) <= imax:
            k = k + 1
        return q[:k]
    
    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)
    
    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x),20)),imax)
    
    if len(q) > 0:
        p = np.abs(q[:,0] / q[:,1] - abs(x)).astype(float) * (1 + np.abs(q[:,0])) * q[:,1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i,0] / float(q[i,1]), xsign* q[i,0], q[i,1], p[i])
    else:
        return (None, 0, 0, 1)

def integerSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p - np.round(p.astype(np.double)))
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.round(p.astype(np.double))[chosen])))

def zeroSnap(p, top=1):
    p = np.array(p)
    metric = np.abs(p)
    chosen = np.argsort(metric)[:top]
    return dict(list(zip(chosen, np.zeros(len(chosen)))))

def rationalSnap(p, top=1):
    """Snap to nearest rational number using continued fraction."""
    p = np.array(p)
    snaps = np.array(list(bestApproximation(x,10) for x in p))
    chosen = np.argsort(snaps[:, 3])[:top]    
    d = dict(list(zip(chosen, snaps[chosen, 1:3])))
    d = {k:  f"{val[0]}/{val[1]}" for k,val in d.items()}
    
    return d