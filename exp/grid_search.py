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


def split_partition(partition):
    block_large = partition[0]
    block_others = partition[1:]
    result = []
    for part in itertools.combinations(block_large, len(block_large) - 1):
        A = list(part)
        B = [x for x in block_large if x not in A]
        result.append([A, B] + block_others)
        
        for i, block in enumerate(block_others):
            merged = sorted(B + block)
            other_remaining = [block_others[j] for j in range(len(block_others)) if j != i]
            result.append([A, merged] + other_remaining)
    return result


def compute_partition_loss(X, partition, thresholds, priors, threshold_true, c):
    thresholds_p = thresholds[partition]
    priors_p     = priors[partition]
    X_p = best_response_vectorized(X, thresholds_p, priors_p, c)
    return accuracy_loss_vectorized(X, X_p, thresholds_p, priors_p, threshold_true)


def compute_system_loss(X, partitions, thresholds, priors, threshold_true, c):
    acc_loss = 0.0
    for partition in partitions:
        acc_loss_p = compute_partition_loss(X, partition, thresholds, priors, threshold_true, c)
        acc_loss += acc_loss_p * np.sum(priors[partition])
    return acc_loss


def find_partitions_optimal(X, thresholds, priors, threshold_true, c):
    indices = list(range(len(thresholds)))
    best_partition, best_loss = None, np.inf
    for partitions in set_partitions(indices):
        acc_loss = compute_system_loss(X, partitions, thresholds, priors, threshold_true, c)
        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = partitions
    return best_partition


def find_partitions_greedy_agglomerative(X, X_eps, thresholds, priors, threshold_true, c, eps=1e-9):
    P = {}
    next_id = 0
    for i in range(len(priors)):
        P[next_id] = [i]
        next_id += 1

    pq = []
    for a_id, b_id in itertools.combinations(P.keys(), 2):
        a, b = P[a_id], P[b_id]
        ab = sorted(a+b)
        acc_loss_ab  = compute_partition_loss(X, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab])
        acc_loss_a   = compute_partition_loss(X, a, thresholds, priors, threshold_true, c)  * np.sum(priors[a])
        acc_loss_b   = compute_partition_loss(X, b, thresholds, priors, threshold_true, c)  * np.sum(priors[b])
        acc_loss_ab += compute_partition_loss(X_eps, ab, thresholds, priors, threshold_true, c) * np.sum(priors[ab]) * eps
        acc_loss_a  += compute_partition_loss(X_eps, a,  thresholds, priors, threshold_true, c) * np.sum(priors[a])  * eps
        acc_loss_b  += compute_partition_loss(X_eps, b,  thresholds, priors, threshold_true, c) * np.sum(priors[b])  * eps
        gain = -(acc_loss_a + acc_loss_b - acc_loss_ab)
        heapq.heappush(pq, (gain, (a_id, b_id)))

    while pq:
        gain, (a_id, b_id) = heapq.heappop(pq)
        if a_id not in P or b_id not in P:
            continue
        a, b = P[a_id], P[b_id]
        ab = sorted(a + b)

        acc_loss_a   = compute_partition_loss(X, a, thresholds, priors, threshold_true, c)
        acc_loss_b   = compute_partition_loss(X, b, thresholds, priors, threshold_true, c)
        acc_loss_ab  = compute_partition_loss(X, ab, thresholds, priors, threshold_true, c)
        acc_loss_a  += compute_partition_loss(X_eps, a,  thresholds, priors, threshold_true, c) * eps
        acc_loss_b  += compute_partition_loss(X_eps, b,  thresholds, priors, threshold_true, c) * eps
        acc_loss_ab += compute_partition_loss(X_eps, ab, thresholds, priors, threshold_true, c) * eps

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
                acc_loss_merged = compute_partition_loss(X, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged])
                acc_loss_p      = compute_partition_loss(X, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])
                acc_loss_ab_new = compute_partition_loss(X, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])
                acc_loss_merged += compute_partition_loss(X_eps, merged, thresholds, priors, threshold_true, c) * np.sum(priors[merged]) * eps
                acc_loss_p      += compute_partition_loss(X_eps, p,      thresholds, priors, threshold_true, c) * np.sum(priors[p])      * eps
                acc_loss_ab_new += compute_partition_loss(X_eps, ab,     thresholds, priors, threshold_true, c) * np.sum(priors[ab])     * eps
                gain = -(acc_loss_p + acc_loss_ab_new - acc_loss_merged)
                heapq.heappush(pq, (gain, (new_id, p_id)))
    return list(P.values())


def find_partitions_greedy_divisive(X, X_eps, thresholds, priors, threshold_true, c, eps=1e-9):
    n = len(priors)
    
    def block_cost(block):
        return (compute_partition_loss(X, block, thresholds, priors, threshold_true, c)
                + compute_partition_loss(X_eps, block, thresholds, priors, threshold_true, c) * eps)

    partition_0 = [list(range(n))]
    acc_loss_0 = block_cost(partition_0[0])
    pq = [(acc_loss_0, partition_0)]

    while pq:
        acc_loss_merged, partition_merged = heapq.heappop(pq)
        if len(partition_merged[0])==1:
            return partition_merged
        partitions = split_partition(partition_merged)
        pq = []
        for partition in partitions:
            acc_loss_split = 0.
            for block in partition:
                acc_loss_split += block_cost(block) * np.sum(priors[block])
            gain = acc_loss_merged - acc_loss_split
            if  gain > 0:
                heapq.heappush(pq, (acc_loss_split, partition))
        if not pq:
            return partition_merged

def approximation_ratio(loss_optimal, loss_greedy, rtype="m"):
    if rtype in ["a", "add", "additive"]:
        return loss_greedy - loss_optimal
    elif rtype in ["m", "mult", "multiplicative"]:
        if loss_optimal == 0:
            return np.nan
        return loss_greedy / loss_optimal


def generate_prior_grid(n_components, n_balls=100, n_balls_eps=100, eps=0.1, p0=0.3):
    assert n_components >= 3, "need at least 3 classifiers"

    eps_step = eps / n_balls_eps
    eps_splits = [(k * eps_step, (n_balls_eps - k) * eps_step)
                  for k in range(n_balls_eps + 1)]

    n_rem = n_components - 3
    rem_total = 1.0 - p0 - eps
    if n_rem == 0:
        rem_splits = [()]
    else:
        rem_step = rem_total / n_balls
        rem_splits = [tuple(np.bincount(combo, minlength=n_rem) * rem_step)
                      for combo in itertools.combinations_with_replacement(range(n_rem), n_balls)]

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

    n_balls = 100
    n_balls_eps = 100
    thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_grid = generate_prior_grid(6, n_balls=n_balls, eps=0.04, n_balls_eps=100)
    threshold_true = 0.2
    c = 0.75
    X_eps = np.arange(-1 / c, 0, 1e-4).round(4)

    partition_base = [[i] for i in range(N_THRESHOLDS)]
    
    results = []
    for priors in tqdm.tqdm(prior_grid, total=len(prior_grid)):
        priors = np.array(priors)

        partition_greedy_agg = find_partitions_greedy_agglomerative(X, X_eps, thresholds, priors, threshold_true, c)
        partition_greedy_div = find_partitions_greedy_divisive(X, X_eps, thresholds, priors, threshold_true, c)
        partition_opt = find_partitions_optimal(X, thresholds, priors, threshold_true, c)
        
        loss_base = compute_system_loss(X, partition_base, thresholds, priors, threshold_true, c)
        loss_opt = compute_system_loss(X, partition_opt, thresholds, priors, threshold_true, c)
        loss_greedy_agg = compute_system_loss(X, partition_greedy_agg, thresholds, priors, threshold_true, c)
        loss_greedy_div = compute_system_loss(X, partition_greedy_div, thresholds, priors, threshold_true, c)

        r_mult_agg = approximation_ratio(loss_opt, loss_greedy_agg, rtype="m")
        r_mult_div = approximation_ratio(loss_opt, loss_greedy_div, rtype="m")
        r_add_agg = approximation_ratio(loss_opt, loss_greedy_agg, rtype="a")
        r_add_div = approximation_ratio(loss_opt, loss_greedy_div, rtype="a")

        result = {
            "c": c,
            "threshold_true": threshold_true,
            "thresholds": thresholds,
            "priors": priors,
            "partition_base": partition_base,
            "partition_opt": partition_opt,
            "partition_greedy_agg": partition_greedy_agg,
            "partition_greedy_div": partition_greedy_div,
            "loss_base": loss_base,
            "loss_opt": loss_opt,
            "loss_greedy_agg": loss_greedy_agg,
            "loss_greedy_div": loss_greedy_div,
            "r_mult_agg": r_mult_agg,
            "r_mult_div": r_mult_div,
            "r_add_agg": r_add_agg,
            "r_add_div": r_add_div
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    print(df_results.shape)

    filepath = f"grid_search_n{N_THRESHOLDS}_refined.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(df_results, f)
    print(f"Saved to {filepath}")
