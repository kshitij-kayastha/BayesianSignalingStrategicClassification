"""
Optimized grid search for approximation ratio.

Optimizations over the notebook version:
1. Partition enumeration is done once per (n, thresholds) config, not per (c, threshold_true) pair.
2. best_response_vectorized is precomputed for each (block, c) pair since it doesn't depend on
   threshold_true -- reducing calls by a factor of len(thresholds_true).
3. The (threshold_true, c) grid is parall elized with joblib.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tqdm
import pickle
from copy import deepcopy
import collections
import itertools
from joblib import Parallel, delayed

from src.util import set_partitions
from src.subroutine import best_response_vectorized, accuracy_loss_vectorized
from src.metric import approximation_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_weighted_loss(block_tuple, X, thresholds, priors, Xp_cache_c, threshold_true):
    """Weighted accuracy loss for a block, using precomputed X_p for fixed c."""
    block = np.array(block_tuple)
    thresholds_p = thresholds[block]
    priors_p = priors[block]
    X_p = Xp_cache_c[block_tuple]
    acc_loss_p = accuracy_loss_vectorized(X, X_p, thresholds_p, priors_p, threshold_true)
    return float(acc_loss_p * np.sum(priors_p))


# ---------------------------------------------------------------------------
# Optimized optimal partition search
# ---------------------------------------------------------------------------

def find_partitions_optimal_fast(all_partitions, unique_blocks, X, thresholds, priors, Xp_cache_c, threshold_true):
    """
    Find the optimal partition using precomputed X_p values (fixed c).

    Parameters
    ----------
    all_partitions : list[list[list[int]]]
        All set partitions (precomputed once outside the grid loop).
    unique_blocks : list[tuple[int]]
        All unique blocks across all partitions.
    X : np.ndarray, shape (N,)
        Original feature vector.
    thresholds, priors : np.ndarray, shape (n,)
    Xp_cache_c : dict[block_tuple -> np.ndarray]
        Precomputed best responses for the current c value.
    threshold_true : float
    """
    block_loss = {
        block_tuple: _block_weighted_loss(block_tuple, X, thresholds, priors, Xp_cache_c, threshold_true)
        for block_tuple in unique_blocks
    }

    best_partition = None
    best_loss = np.inf

    for partitions in all_partitions:
        acc_loss = sum(block_loss[tuple(sorted(block))] for block in partitions)
        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = deepcopy(partitions)

    return best_partition, best_loss


# ---------------------------------------------------------------------------
# Optimized greedy partition search
# ---------------------------------------------------------------------------

def find_partitions_greedy_fast(X, thresholds, priors, Xp_cache_c, threshold_true):
    """
    Greedy partition search using precomputed X_p for fixed c.

    Parameters
    ----------
    X : np.ndarray, shape (N,)
        Original feature vector.
    thresholds, priors : np.ndarray, shape (n,)
    Xp_cache_c : dict[block_tuple -> np.ndarray]
        Precomputed best responses for the current c value.
    threshold_true : float
    """
    n = len(thresholds)
    P = {i: (i,) for i in range(n)}  # id -> block_tuple
    next_id = n

    Q = collections.deque(itertools.combinations(range(n), 2))

    loss_cache = {}

    def get_loss(block_tuple):
        if block_tuple not in loss_cache:
            loss_cache[block_tuple] = _block_weighted_loss(
                block_tuple, X, thresholds, priors, Xp_cache_c, threshold_true
            )
        return loss_cache[block_tuple]

    while Q:
        a_id, b_id = Q.popleft()
        if a_id not in P or b_id not in P:
            continue

        a = P[a_id]
        b = P[b_id]
        ab = tuple(sorted(a + b))

        lhs = get_loss(a) + get_loss(b)
        rhs = get_loss(ab)

        if lhs - rhs > -1e-6:
            del P[a_id]
            del P[b_id]

            Q = collections.deque(
                (x, y) for (x, y) in Q
                if x not in {a_id, b_id} and y not in {a_id, b_id}
            )

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for c_id in P:
                if c_id != new_id:
                    Q.append((new_id, c_id))

    return [list(block) for block in P.values()]


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def precompute_Xp(X, thresholds, priors, unique_blocks, C):
    """
    Precompute best responses for all (block, c) pairs.

    best_response_vectorized only depends on c (not threshold_true), so computing
    it here once saves a factor of len(thresholds_true) evaluations.

    Returns
    -------
    dict[(block_tuple, c_float) -> np.ndarray of shape (len(X),)]
    """
    Xp_cache = {}
    for block_tuple in unique_blocks:
        block = np.array(block_tuple)
        thresholds_p = thresholds[block]
        priors_p = priors[block]
        for c in C:
            Xp_cache[(block_tuple, float(c))] = best_response_vectorized(
                X, thresholds_p, priors_p, float(c)
            )
    return Xp_cache


# ---------------------------------------------------------------------------
# Single grid-point worker (called by joblib)
# ---------------------------------------------------------------------------

def run(threshold_true, c, all_partitions, unique_blocks, X, thresholds, priors, Xp_cache_c):
    """
    Evaluate one (threshold_true, c) grid point.

    Xp_cache_c : dict[block_tuple -> X_p] sliced to this c value.
    """
    partition_opt, acc_loss_opt = find_partitions_optimal_fast(
        all_partitions, unique_blocks, X, thresholds, priors, Xp_cache_c, threshold_true
    )

    partition_greedy = find_partitions_greedy_fast(
        X, thresholds, priors, Xp_cache_c, threshold_true
    )

    acc_loss_greedy = sum(
        _block_weighted_loss(
            tuple(sorted(block)), X, thresholds, priors, Xp_cache_c, threshold_true
        )
        for block in partition_greedy
    )

    r_mult = approximation_ratio(acc_loss_opt, acc_loss_greedy, rtype="mult")
    r_add = approximation_ratio(acc_loss_opt, acc_loss_greedy, rtype="add")

    return {
        "threshold_true": float(threshold_true),
        "c": float(c),
        "partition_opt": partition_opt,
        "partition_greedy": partition_greedy,
        "acc_loss_opt": float(acc_loss_opt),
        "acc_loss_greedy": float(acc_loss_greedy),
        "r_mult": r_mult,
        "r_add": r_add,
    }

def generate_prior_grid(n_components=3, n_balls=50):
    step = 1.0 / n_balls
    grids = []
    for combo in itertools.combinations_with_replacement(range(n_components), n_balls):
        counts = np.bincount(combo, minlength=n_components).round(5)
        grids.append(counts * step)
    return grids

def generate_threshold_grid(n_components=3, step=0.02):
    ticks = np.arange(0, 1 + step, step).round(5)
    return [np.array(combo) for combo in itertools.combinations(ticks, n_components)]

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(X, thresholds, priors, C, thresholds_true, n_jobs=-1):
    """
    Optimized grid search over (threshold_true, c) pairs for fixed thresholds/priors.

    Parameters
    ----------
    X : np.ndarray, shape (N,)
    thresholds : np.ndarray, shape (n,)
    priors : np.ndarray, shape (n,)
    C : array-like of float
    thresholds_true : array-like of float
    n_jobs : int
        Passed to joblib.Parallel (-1 = all cores).

    Returns
    -------
    pd.DataFrame with columns: threshold_true, c, partition_opt, partition_greedy,
                                acc_loss_opt, acc_loss_greedy, ratio
    """
    n = len(thresholds)
    indices = list(range(n))

    # --- Step 1: enumerate all partitions once ---
    all_partitions = list(set_partitions(indices))
    # Collect every unique block that can appear (from enumerated partitions + greedy merges)
    # For greedy we also need merged blocks, so include all subsets
    unique_blocks_set = {
        tuple(sorted(combo))
        for size in range(1, n + 1)
        for combo in itertools.combinations(indices, size)
    }
    unique_blocks = list(unique_blocks_set)

    # --- Step 2: precompute X_p for each (block, c) ---
    Xp_full = precompute_Xp(X, thresholds, priors, unique_blocks, C)

    # --- Step 3: build per-c slices to pass to workers ---
    tasks = []
    for threshold_true in thresholds_true:
        for c in C:
            Xp_cache_c = {block: Xp_full[(block, c)] for block in unique_blocks}
            tasks.append((float(threshold_true), c, Xp_cache_c))

    # --- Step 4: run grid in parallel ---
    rows = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run)(t, c, all_partitions, unique_blocks,
                          X, thresholds, priors, Xp_c)
        for t, c, Xp_c in tasks
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main: reproduce random-n grid from the notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)
    N_THRESHOLDS = 3

    N_THRESHOLDS = 3
    step = 0.05
    step_tt = 0.1
    n_balls = 50
    threshold_grid = generate_threshold_grid(n_components=N_THRESHOLDS, step=step)
    prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls)
    threshold_true_grid = np.arange(step_tt, 1+step_tt, step_tt).round(5)
    c_grid = [0.75]

    all_results = []

    for thresholds, priors in tqdm.tqdm(itertools.product(threshold_grid, prior_grid), total=len(threshold_grid)*len(prior_grid)):
        df = grid_search(
            X, thresholds, priors,
            C=c_grid,
            thresholds_true=threshold_true_grid,
            n_jobs=-1,
        )
        df["thresholds"] = [deepcopy(thresholds)] * len(df)
        df["priors"] = [deepcopy(priors)] * len(df)
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    print(results_df.shape)
    print(results_df.head())

    with open("grid_search_approx_ratio_fast.pkl", "wb") as f:
        pickle.dump(results_df, f)
    print("Saved to grid_search_approx_ratio_fast.pkl")
