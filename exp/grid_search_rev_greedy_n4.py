import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tqdm
import pickle
from copy import deepcopy
import heapq
import itertools
from joblib import Parallel, delayed

from src.subroutine import best_response_vectorized, accuracy_loss_vectorized


def set_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]
        yield [[first]] + smaller


def split_partition(partition, full_split=False):
    block_large = partition[0]
    block_others = partition[1:]
    result = []
    for part in itertools.combinations(block_large, len(block_large) - 1):
        A = list(part)
        B = [x for x in block_large if x not in A]
        result.append([A, B] + block_others)
        if full_split:
            for i, block in enumerate(block_others):
                merged = sorted(B + block)
                other_remaining = [block_others[j] for j in range(len(block_others)) if j != i]
                result.append([A, merged] + other_remaining)
    return result



def precompute_Xp(X, X_eps, thresholds, priors, unique_blocks, c):
    Xp_cache, Xp_eps_cache = {}, {}
    for block_tuple in unique_blocks:
        block = np.array(block_tuple)
        thresholds_p = thresholds[block]
        priors_p = priors[block]
        Xp_cache[block_tuple] = best_response_vectorized(X, thresholds_p, priors_p, float(c))
        Xp_eps_cache[block_tuple] = best_response_vectorized(X_eps, thresholds_p, priors_p, float(c))
    return Xp_cache, Xp_eps_cache


def make_block_loss(X, X_eps, thresholds, priors, Xp_cache, Xp_eps_cache, threshold_true, eps=1e-9):
    cache = {}
    def fn(block_tuple):
        if block_tuple not in cache:
            block = np.array(block_tuple)
            thresholds_p = thresholds[block]
            priors_p = priors[block]
            w = float(np.sum(priors_p))
            l = float(accuracy_loss_vectorized(
                X, Xp_cache[block_tuple], thresholds_p, priors_p, threshold_true))
            l_eps = float(accuracy_loss_vectorized(
                X_eps, Xp_eps_cache[block_tuple], thresholds_p, priors_p, threshold_true))
            cache[block_tuple] = (l + eps * l_eps) * w
        return cache[block_tuple]
    return fn


def compute_system_loss(partitions, X, thresholds, priors, Xp_cache, threshold_true):
    total = 0.0
    for partition in partitions:
        block_tuple = tuple(sorted(partition))
        block = np.array(block_tuple)
        thresholds_p = thresholds[block]
        priors_p = priors[block]
        l = float(accuracy_loss_vectorized(
            X, Xp_cache[block_tuple], thresholds_p, priors_p, threshold_true))
        total += l * float(np.sum(priors_p))
    return total


def find_partitions_optimal(all_partitions, block_loss):
    best_partition, best_loss = None, np.inf
    for partition in all_partitions:
        loss = sum(block_loss(tuple(sorted(b))) for b in partition)
        if loss < best_loss:
            best_loss = loss
            best_partition = deepcopy(partition)
    return best_partition


def find_partitions_greedy_agglomerative(initial_partition, block_loss):
    P = {i: tuple(sorted(b)) for i, b in enumerate(initial_partition)}
    next_id = len(initial_partition)
    counter = itertools.count()

    pq = []
    for a_id, b_id in itertools.combinations(list(P.keys()), 2):
        a, b = P[a_id], P[b_id]
        ab = tuple(sorted(a + b))
        gain = -(block_loss(a) + block_loss(b) - block_loss(ab))
        heapq.heappush(pq, (gain, next(counter), a_id, b_id))

    while pq:
        gain, _, a_id, b_id = heapq.heappop(pq)
        if a_id not in P or b_id not in P:
            continue
        a, b = P[a_id], P[b_id]
        ab = tuple(sorted(a + b))

        if block_loss(a) + block_loss(b) - block_loss(ab) > -1e-9:
            del P[a_id]
            del P[b_id]
            pq = [(g, c, x, y) for g, c, x, y in pq
                  if x not in {a_id, b_id} and y not in {a_id, b_id}]
            heapq.heapify(pq)

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for p_id, p in P.items():
                if p_id == new_id:
                    continue
                abc = tuple(sorted(ab + p))
                gain = -(block_loss(ab) + block_loss(p) - block_loss(abc))
                heapq.heappush(pq, (gain, next(counter), new_id, p_id))

    return [list(b) for b in P.values()]


def find_partitions_greedy_divisive(n, block_loss, full_split=False):
    def partition_cost(partition):
        return sum(block_loss(tuple(sorted(b))) for b in partition)

    counter = itertools.count()
    partition_0 = [list(range(n))]
    pq = [(partition_cost(partition_0), next(counter), partition_0)]

    while pq:
        cost_merged, _, partition_merged = heapq.heappop(pq)
        if len(partition_merged[0]) == 1:
            return partition_merged
        splits = split_partition(partition_merged, full_split)
        pq = []
        for partition in splits:
            cost_split = partition_cost(partition)
            if cost_merged - cost_split >= 0:
                heapq.heappush(pq, (cost_split, next(counter), partition))
        if not pq:
            return partition_merged

def approximation_ratio(loss_optimal, loss_greedy, rtype="m"):
    if rtype in ["a", "add", "additive"]:
        return loss_greedy - loss_optimal
    elif rtype in ["m", "mult", "multiplicative"]:
        if loss_optimal == 0:
            return np.nan
        return loss_greedy / loss_optimal


def run(threshold_true, all_partitions, n, X, X_eps, thresholds, priors, Xp_cache, Xp_eps_cache):
    block_loss = make_block_loss(X, X_eps, thresholds, priors, Xp_cache, Xp_eps_cache,
                                  float(threshold_true))

    partition_base = [[i] for i in range(n)]
    partition_opt = find_partitions_optimal(all_partitions, block_loss)
    partition_greedy_agg = find_partitions_greedy_agglomerative([[i] for i in range(n)], block_loss)
    partition_greedy_div = find_partitions_greedy_divisive(n, block_loss, full_split=True)

    def loss(p):
        return compute_system_loss(p, X, thresholds, priors, Xp_cache, float(threshold_true))

    loss_base = loss(partition_base)
    loss_opt = loss(partition_opt)
    loss_greedy_agg = loss(partition_greedy_agg)
    loss_greedy_div = loss(partition_greedy_div)

    r_mult_agg = approximation_ratio(loss_opt, loss_greedy_agg)
    r_add_agg = approximation_ratio(loss_opt, loss_greedy_agg, "a")
    r_mult_div = approximation_ratio(loss_opt, loss_greedy_div)
    r_add_div = approximation_ratio(loss_opt, loss_greedy_div, "a")

    return {
        "threshold_true": float(threshold_true),
        "partition_base": partition_base,
        "partition_opt": partition_opt,
        "partition_greedy_agg": partition_greedy_agg,
        "partition_greedy_div": partition_greedy_div,
        "loss_base": loss_base,
        "loss_opt": loss_opt,
        "loss_greedy_agg": loss_greedy_agg,
        "loss_greedy_div": loss_greedy_div,
        "r_mult_agg": r_mult_agg,
        "r_add_agg": r_add_agg,
        "r_mult_div": r_mult_div,
        "r_add_div": r_add_div
    }


# def generate_prior_grid(n_components=3, n_balls=50):
#     step = 1.0 / n_balls
#     grids = []
#     for combo in itertools.combinations_with_replacement(range(n_components), n_balls):
#         counts = np.bincount(combo, minlength=n_components).round(5)
#         grids.append(counts * step)
#     return grids

def generate_prior_grid(n_components, n_balls=50, n_balls_eps=10, eps=0.01, p0=0.3):
    """
    Prior vectors of length n_components where:
      prior[0]            = p0                      (fixed, default 0.3)
      prior[1] + prior[2] = eps                     (split in n_balls_eps steps)
      sum(prior[3:])      = 1 - p0 - eps            (stars-and-bars, n_balls steps)
    """
    assert n_components >= 3, "need at least 3 classifiers"

    # --- split eps across classifiers 1 and 2 ---
    eps_step = eps / n_balls_eps
    eps_splits = [(k * eps_step, (n_balls_eps - k) * eps_step)
                  for k in range(n_balls_eps + 1)]

    # --- split the remaining mass across classifiers 3 .. n-1 ---
    n_rem = n_components - 3
    rem_total = 1.0 - p0 - eps
    if n_rem == 0:
        rem_splits = [()]                  # nothing left to distribute
    else:
        rem_step = rem_total / n_balls
        rem_splits = [tuple(np.bincount(combo, minlength=n_rem) * rem_step)
                      for combo in itertools.combinations_with_replacement(range(n_rem), n_balls)]

    # --- cartesian product of the two sub-grids ---
    grids = []
    for e1, e2 in eps_splits:
        for rem in rem_splits:
            grids.append(np.array((p0, e1, e2) + rem))
    return grids


def generate_threshold_grid(n_components=3, step=0.02):
    ticks = np.arange(0, 1 + step, step).round(5)
    return [np.array(combo) for combo in itertools.combinations(ticks, n_components)]


if __name__ == "__main__":
    np.random.seed(0)

    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)
    N_THRESHOLDS = 6

    # step = 0.2
    # step_tt = 0.1
    n_balls = 100
    # threshold_grid = generate_threshold_grid(n_components=N_THRESHOLDS, step=step)
    # threshold_grid = [np.array([0.0, 0.2, 0.4, 0.6, 1.0]), np.array([0.0, 0.2, 0.4, 0.6, 0.8]), np.array([0.0, 0.2, 0.4, 0.8, 1.0])]
    threshold_grid = [np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]
    # prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls)
    prior_grid = generate_prior_grid(6, n_balls=n_balls, eps=0.04, n_balls_eps=100)
    # threshold_true_grid = np.arange(step_tt, 1 + step_tt, step_tt).round(5)
    threshold_true_grid = [0.2]
    c_grid = [0.75]

    indices = list(range(N_THRESHOLDS))
    all_partitions = list(set_partitions(indices))
    unique_blocks = list({
        tuple(sorted(combo))
        for size in range(1, N_THRESHOLDS + 1)
        for combo in itertools.combinations(indices, size)
    })

    all_results = []

    for thresholds, priors in tqdm.tqdm(
        itertools.product(threshold_grid, prior_grid),
        total=len(threshold_grid) * len(prior_grid),
    ):
        thresholds = np.array(thresholds)
        priors = np.array(priors)

        for c in c_grid:
            X_eps = np.arange(-1 / c, 0, 1e-3).round(5)
            Xp_cache, Xp_eps_cache = precompute_Xp(X, X_eps, thresholds, priors, unique_blocks, c)

            rows = Parallel(n_jobs=-1, verbose=0)(
                delayed(run)(
                    threshold_true, all_partitions, N_THRESHOLDS,
                    X, X_eps, thresholds, priors, Xp_cache, Xp_eps_cache,
                )
                for threshold_true in threshold_true_grid
            )
            for row in rows:
                row["c"] = float(c)
                row["thresholds"] = deepcopy(thresholds)
                row["priors"] = deepcopy(priors)
            all_results.extend(rows)

    results_df = pd.DataFrame(all_results)
    print(results_df.shape)

    filepath = f"grid_search_n{N_THRESHOLDS}_refined.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(results_df, f)
    print(f"Saved to {filepath}")
