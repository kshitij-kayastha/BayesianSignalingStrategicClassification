"""
Optimized grid search for approximation ratio.

Optimizations over the notebook version:
1. Partition enumeration is done once per (n, thresholds) config, not per (c, threshold_true) pair.
2. best_response_vectorized is precomputed for each (block, c) pair since it doesn't depend on
   threshold_true -- reducing calls by a factor of len(thresholds_true).
3. The (threshold_true, c) grid is parall elized with joblib.
"""

import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import tqdm
import pickle
from copy import deepcopy
import itertools

from src.subroutine import evaluate_system
from src.metric import approximation_ratio
from src.algorithm import find_partitions_greedy_best

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
# Main: reproduce random-n grid from the notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)

    pkl_path = os.path.join(os.path.dirname(__file__), "results", "grid_search_approx_ratio_fast.pkl")
    with open(pkl_path, "rb") as f:
        results_df = pickle.load(f)
    print(f"Loaded {len(results_df)} rows from {pkl_path}")

    rows = []
    for _, row in tqdm.tqdm(results_df.iterrows(), total=len(results_df)):
        thresholds = np.array(row["thresholds"])
        priors = np.array(row["priors"])
        threshold_true = float(row["threshold_true"])
        c = float(row["c"])

        partition_greedy = find_partitions_greedy_best(X, thresholds, priors, threshold_true, c)

        acc_loss_greedy = evaluate_system(X, partition_greedy, thresholds, priors, threshold_true, c)
        acc_loss_opt = float(row["acc_loss_opt"])

        r_mult = approximation_ratio(acc_loss_opt, acc_loss_greedy, rtype="mult")
        r_add = approximation_ratio(acc_loss_opt, acc_loss_greedy, rtype="add")

        rows.append({
            "threshold_true": threshold_true,
            "c": c,
            "thresholds": deepcopy(thresholds),
            "priors": deepcopy(priors),
            "partition_opt": row["partition_opt"],
            "partition_greedy": partition_greedy,
            "acc_loss_opt": acc_loss_opt,
            "acc_loss_greedy": float(acc_loss_greedy),
            "r_mult": r_mult,
            "r_add": r_add,
        })

    out_df = pd.DataFrame(rows)
    print(out_df.shape)
    print(out_df.head())

    out_path = "grid_search_approx_ratio_best.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out_df, f)
    print(f"Saved to {out_path}")
