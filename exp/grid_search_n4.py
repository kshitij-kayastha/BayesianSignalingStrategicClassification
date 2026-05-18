"""
Approximation ratio evaluation using greedy-best (priority-queue greedy).

Loads existing results from results/grid_search_approx_ratio_fast.pkl (which contains
opt and standard greedy baselines) and runs find_partitions_greedy_best_fast for each row.

Key optimizations:
1. Rows are grouped by (thresholds, priors, c) so best responses are precomputed once per group,
   not once per (threshold_true) row -- saving a factor of len(thresholds_true) evaluations.
2. Within each group, (threshold_true) tasks are parallelized with joblib.
3. The priority-queue greedy always merges the most beneficial pair first.
"""

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

from src.subroutine import evaluate_partition, evaluate_system

def set_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i+1:]
        yield [[first]] + smaller

def find_partitions_optimal(X, thresholds, priors, threshold_true, c):
    indices = list(range(len(thresholds)))
    best_partition, best_loss = None, np.inf
    for partitions in set_partitions(indices):
        acc_loss = evaluate_system(X, partitions, thresholds, priors, threshold_true, c)
        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = partitions
    return best_partition

def find_partitions_greedy_agglomerative(X, thresholds, priors, threshold_true, c, eps=1e-9):
    P = {}
    next_id = 0
    for i in range(len(priors)):
        P[next_id] = [i]
        next_id += 1

    X_eps = np.arange(-1/c, 0, 1e-3).round(5)

    pq = []
    for a_id, b_id in itertools.combinations(P.keys(), 2):
        a, b = P[a_id], P[b_id]
        ab = sorted(a+b)
        acc_loss_ab = evaluate_partition(X, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab])
        acc_loss_a  = evaluate_partition(X, a, thresholds, priors, threshold_true, c)  * np.sum(priors[a])
        acc_loss_b  = evaluate_partition(X, b, thresholds, priors, threshold_true, c)  * np.sum(priors[b])
        acc_loss_ab += evaluate_partition(X_eps, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab]) * eps
        acc_loss_a  += evaluate_partition(X_eps, a,  thresholds, priors, threshold_true, c) * np.sum(priors[a])  * eps
        acc_loss_b  += evaluate_partition(X_eps, b,  thresholds, priors, threshold_true, c) * np.sum(priors[b])  * eps
        gain = -(acc_loss_a + acc_loss_b - acc_loss_ab)
        heapq.heappush(pq, (gain, (a_id, b_id)))

    while pq:
        gain, (a_id, b_id) = heapq.heappop(pq)
        if a_id not in P or b_id not in P:
            continue
        a, b = P[a_id], P[b_id]
        ab = sorted(a + b)

        acc_loss_a  = evaluate_partition(X, a, thresholds, priors, threshold_true, c)
        acc_loss_b  = evaluate_partition(X, b, thresholds, priors, threshold_true, c)
        acc_loss_ab = evaluate_partition(X, ab, thresholds, priors, threshold_true, c)
        acc_loss_a  += evaluate_partition(X_eps, a,  thresholds, priors, threshold_true, c) * eps
        acc_loss_b  += evaluate_partition(X_eps, b,  thresholds, priors, threshold_true, c) * eps
        acc_loss_ab += evaluate_partition(X_eps, ab, thresholds, priors, threshold_true, c) * eps

        lhs = acc_loss_a * np.sum(priors[a]) + acc_loss_b * np.sum(priors[b])
        rhs = acc_loss_ab * np.sum(priors[ab])

        if lhs - rhs > -1e-9:
            del P[a_id]
            del P[b_id]
            pq = [(g, (x, y)) for g, (x, y) in pq if x not in {a_id, b_id} and y not in {a_id, b_id}]
            heapq.heapify(pq)

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for p_id in P:
                if p_id == new_id:
                    continue
                p = P[p_id]
                merged = sorted(ab + p)
                acc_loss_merged = evaluate_partition(X, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged])
                acc_loss_p      = evaluate_partition(X, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])
                acc_loss_ab_new = evaluate_partition(X, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])
                acc_loss_merged += evaluate_partition(X_eps, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged]) * eps
                acc_loss_p      += evaluate_partition(X_eps, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])      * eps
                acc_loss_ab_new += evaluate_partition(X_eps, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])     * eps
                gain = -(acc_loss_p + acc_loss_ab_new - acc_loss_merged)
                heapq.heappush(pq, (gain, (new_id, p_id)))
    return list(P.values())


def find_partitions_greedy_forward(X, thresholds, priors, threshold_true, c, partition, eps=1e-9):
    P = {}
    next_id = 0
    for i, p in enumerate(partition):
        P[next_id] = p
        next_id += 1

    X_eps = np.arange(-1/c, 0, 1e-3).round(5)
    
    pq = []
    for a_id, b_id in itertools.combinations(P.keys(), 2):
        a, b = P[a_id], P[b_id]
        ab = sorted(a+b)
        acc_loss_ab = evaluate_partition(X, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab])
        acc_loss_a  = evaluate_partition(X, a, thresholds, priors, threshold_true, c)  * np.sum(priors[a])
        acc_loss_b  = evaluate_partition(X, b, thresholds, priors, threshold_true, c)  * np.sum(priors[b])
        acc_loss_ab += evaluate_partition(X_eps, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab]) * eps
        acc_loss_a  += evaluate_partition(X_eps, a,  thresholds, priors, threshold_true, c) * np.sum(priors[a])  * eps
        acc_loss_b  += evaluate_partition(X_eps, b,  thresholds, priors, threshold_true, c) * np.sum(priors[b])  * eps
        gain = -(acc_loss_a + acc_loss_b - acc_loss_ab)
        heapq.heappush(pq, (gain, (a_id, b_id)))

    while pq:
        gain, (a_id, b_id) = heapq.heappop(pq)
        if a_id not in P or b_id not in P:
            continue
        a, b = P[a_id], P[b_id]
        ab = sorted(a + b)

        acc_loss_a  = evaluate_partition(X, a, thresholds, priors, threshold_true, c)
        acc_loss_b  = evaluate_partition(X, b, thresholds, priors, threshold_true, c)
        acc_loss_ab = evaluate_partition(X, ab, thresholds, priors, threshold_true, c)
        acc_loss_a  += evaluate_partition(X_eps, a,  thresholds, priors, threshold_true, c) * eps
        acc_loss_b  += evaluate_partition(X_eps, b,  thresholds, priors, threshold_true, c) * eps
        acc_loss_ab += evaluate_partition(X_eps, ab, thresholds, priors, threshold_true, c) * eps

        lhs = acc_loss_a * np.sum(priors[a]) + acc_loss_b * np.sum(priors[b])
        rhs = acc_loss_ab * np.sum(priors[ab])

        if lhs - rhs > -1e-9:
            del P[a_id]
            del P[b_id]
            pq = [(g, (x, y)) for g, (x, y) in pq if x not in {a_id, b_id} and y not in {a_id, b_id}]
            heapq.heapify(pq)

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for p_id in P:
                if p_id == new_id:
                    continue
                p = P[p_id]
                merged = sorted(ab + p)
                acc_loss_merged = evaluate_partition(X, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged])
                acc_loss_p      = evaluate_partition(X, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])
                acc_loss_ab_new = evaluate_partition(X, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])
                acc_loss_merged += evaluate_partition(X_eps, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged]) * eps
                acc_loss_p      += evaluate_partition(X_eps, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])      * eps
                acc_loss_ab_new += evaluate_partition(X_eps, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])     * eps
                gain = -(acc_loss_p + acc_loss_ab_new - acc_loss_merged)
                heapq.heappush(pq, (gain, (new_id, p_id)))
    return list(P.values())

def approximation_ratio(loss_optimal, loss_greedy, rtype="m"):
    if rtype in ["a", "add", "additive"]:
        return loss_greedy - loss_optimal
    elif rtype in ["m", "mult", "multiplicative"]:
        if loss_optimal == 0:
            return np.nan
        return loss_greedy / loss_optimal


def split_partition(partition, full_split=False):
    idx_large = 0
    block_large = partition[idx_large]
    block_others = [partition[i] for i in range(len(partition)) if i != idx_large]
    result = []
    for part in itertools.combinations(block_large, len(block_large)-1):
        A = list(part)
        B = [x for x in block_large if x not in A]
        result.append([A] + [B] + block_others)

        if full_split:
            for i, block in enumerate(block_others):
                merged = sorted(B + block)
                other_remaining = [block_others[j] for j in range(len(block_others)) if j != i]
                result.append([A] + [merged] + other_remaining)
    return result


def find_partitions_greedy_divisive(X, thresholds, priors, threshold_true, c, eps=1e-9, full_split=False):
    n = len(priors)
    X_eps = np.arange(-1/c, 0, 1e-3).round(5)
    
    def block_cost(block):
        return (evaluate_partition(X, block, thresholds, priors, threshold_true, c)
                + evaluate_partition(X_eps, block, thresholds, priors, threshold_true, c) * eps)

    partition_0 = [list(range(n))]
    acc_loss_0 = block_cost(partition_0[0])
    pq = [(acc_loss_0, partition_0)]

    while pq:
        acc_loss_merged, partition_merged = heapq.heappop(pq)
        if len(partition_merged[0])==1:
            return partition_merged
        partitions = split_partition(partition_merged, full_split)
        pq = []
        for partition in partitions:
            acc_loss_split = 0.
            for block in partition:
                acc_loss_split += block_cost(block) * np.sum(priors[block])
            gain = acc_loss_merged - acc_loss_split
            if  gain >= 0:
                heapq.heappush(pq, (acc_loss_split, partition))
        if not pq:
            return partition_merged


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

if __name__ == "__main__":
    np.random.seed(0)

    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)
    N_THRESHOLDS = 4

    step = 0.2
    step_tt = 0.1
    n_balls = 50
    threshold_grid = generate_threshold_grid(n_components=N_THRESHOLDS, step=step)
    prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls)
    threshold_true_grid = np.arange(step_tt, 1+step_tt, step_tt).round(5)
    # threshold_true_grid = [0.1]
    c_grid = [0.75]

    all_results = []

    for thresholds, priors in tqdm.tqdm(itertools.product(threshold_grid, prior_grid), total=len(threshold_grid)*len(prior_grid)):
        for threshold_true in threshold_true_grid:
            for c in c_grid:

                partition_base = [[i] for i in range(N_THRESHOLDS)]
                acc_loss_base = evaluate_system(X, partition_base, thresholds, priors, threshold_true, c) 
                
                partition_opt = find_partitions_optimal(X, thresholds, priors, threshold_true, c)
                acc_loss_opt = evaluate_system(X, partition_opt, thresholds, priors, threshold_true, c)

                partition_greedy = find_partitions_greedy_agglomerative(X, thresholds, priors, threshold_true, c)
                acc_loss_greedy = evaluate_system(X, partition_greedy, thresholds, priors, threshold_true, c)

                partition_greedy_div = find_partitions_greedy_divisive(X, thresholds, priors, threshold_true, c)
                acc_loss_greedy_div = evaluate_system(X, partition_greedy_div, thresholds, priors, threshold_true, c)

                partition_greedy_div_plus = find_partitions_greedy_divisive(X, thresholds, priors, threshold_true, c, full_split=True)
                acc_loss_greedy_div_plus = evaluate_system(X, partition_greedy_div_plus, thresholds, priors, threshold_true, c)

                partition_greedy_div_forward = find_partitions_greedy_forward(X, thresholds, priors, threshold_true, c, partition_greedy_div)
                acc_loss_greedy_div_forward = evaluate_system(X, partition_greedy_div_forward, thresholds, priors, threshold_true, c)

                partition_greedy_div_plus_forward = find_partitions_greedy_forward(X, thresholds, priors, threshold_true, c, partition_greedy_div_plus)
                acc_loss_greedy_div_plus_forward = evaluate_system(X, partition_greedy_div_plus_forward, thresholds, priors, threshold_true, c)
                
                all_results.append({
                    "threshold_true": float(threshold_true),
                    "c": float(c),
                    "thresholds": deepcopy(thresholds),
                    "priors": deepcopy(priors),
                    "partition_base": partition_base,
                    "partition_opt": partition_opt,
                    "partition_greedy": partition_greedy,
                    "partition_greedy_div": partition_greedy_div,
                    "partition_greedy_div_plus": partition_greedy_div_plus,
                    "partition_greedy_div_forward": partition_greedy_div_forward,
                    "partition_greedy_div_plus_forward": partition_greedy_div_plus_forward,
                    "acc_loss_base": float(acc_loss_base),
                    "acc_loss_opt": float(acc_loss_opt),
                    "acc_loss_greedy": float(acc_loss_greedy),
                    "acc_loss_greedy_div": float(acc_loss_greedy_div),
                    "acc_loss_greedy_div_plus": float(acc_loss_greedy_div_plus),
                    "acc_loss_greedy_div_forward": float(acc_loss_greedy_div_forward),
                    "acc_loss_greedy_div_plus_forward": float(acc_loss_greedy_div_plus_forward)
                })
        

    results_df = pd.DataFrame(all_results, ignore_index=True)
    print(results_df.shape)
    print(results_df.head())

    with open("grid_search_approx_ratio_n4_divisive.pkl", "wb") as f:
        pickle.dump(results_df, f)
    print("Saved to grid_search_approx_ratio_n4_divisive.pkl")
