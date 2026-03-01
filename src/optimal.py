import collections
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import plotly.express as px
import tqdm
from copy import deepcopy
from functools import lru_cache


np.random.seed(0)


def bayesian_update(priors):
    if np.sum(priors) == 0:
        return np.zeros_like(priors)
    return priors / np.sum(priors)

def best_response(x, thresholds, priors, c):
    posteriors = bayesian_update(priors)
    search_space = [x] + thresholds[thresholds > x].tolist()
    utilities = []
    for x_p in search_space:
        utility = np.dot(posteriors, x_p >= thresholds)
        cost = c * abs(x-x_p)
        utilities.append(utility - cost)
    return search_space[np.argmax(utilities)]

def merge_classifiers(X, X_p, thresholds, priors):
    merged_thresholds, merged_priors = deepcopy(thresholds), deepcopy(priors)
    for _ in range(len(thresholds)):
        support = np.array([np.mean(X_p==threshold) for threshold in merged_thresholds])
        dominated = np.where(support == 0)[0]

        if len(dominated) == 0:
            break
        
        for i in reversed(dominated):
            threshold_i = merged_thresholds[i]

            jumps = X_p[X < threshold_i]
            if len(jumps) == 0:
                continue

            candidates = jumps[jumps > threshold_i]
            if len(candidates) == 0:
                continue

            values, counts = np.unique(candidates, return_counts=True)
            threshold_dom = values[np.argmax(counts)]

            dom_idx = np.where(threshold_dom == merged_thresholds)[0][0]

            merged_priors[dom_idx] += merged_priors[i]
            merged_priors = np.delete(merged_priors, i)
            merged_thresholds = np.delete(merged_thresholds, i)

    return merged_thresholds, merged_priors


def accuracy_loss(X, X_p, thresholds, priors, threshold_true):
    Y_true = (X >= threshold_true).astype(float)
    losses = []
    for threshold in thresholds:
        Y_p = (X_p >= threshold).astype(float)
        acc_loss = np.abs(Y_true - Y_p).mean()
        losses.append(acc_loss)
    losses = np.array(losses)
    posteriors = bayesian_update(priors)
    return np.dot(losses, posteriors)

@lru_cache(maxsize=None)
def evaluate_partition(X, partition, thresholds, priors, threshold_true, c):
    thresholds_p = thresholds[partition]
    priors_p = priors[partition]
    X_p = np.array([best_response(x, thresholds_p, priors_p, c) for x in X])
    thresholds_p, priors_p = merge_classifiers(X, X_p, thresholds_p, priors_p)
    acc_loss_p = accuracy_loss(X, X_p, thresholds_p, priors_p, threshold_true)
    return acc_loss_p

def evaluate_system(X, partitions, thresholds, priors, threshold_true, c):
    acc_loss = 0.
    for partition in partitions:
        acc_loss_p = evaluate_partition(X, partition, thresholds, priors, threshold_true, c)
        acc_loss += acc_loss_p * np.sum(priors[partition])
    return acc_loss


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
    partitions_set = list(set_partitions(indices))

    @lru_cache(maxsize=None)
    def evaluate_partition_cached(partition_tuple):
        partition = np.array(partition_tuple)
        acc_loss_p = evaluate_partition(
            X, partition, thresholds, priors, threshold_true, c
        )
        return acc_loss_p * np.sum(priors[partition])

    best_partition = None
    best_loss = np.inf

    for partitions in tqdm.tqdm(partitions_set):

        acc_loss = 0.0
        for partition in partitions:
            partition_tuple = tuple(sorted(partition))  # must be hashable
            acc_loss += evaluate_partition_cached(partition_tuple)

        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = deepcopy(partitions)

    return best_partition


if __name__ == "__main__":
    _GLOBALS = {}
    c = 5.0
    threshold_true = 0.5

    threshold_min, threshold_max, threshold_delta = 0., 1., 0.1
    thresholds = np.arange(threshold_min+threshold_delta, threshold_max, threshold_delta).round(4)
    X = np.arange(threshold_min, threshold_max, 1e-4).round(4)

    # priors = np.zeros_like(thresholds)
    priors = np.array([0.11393634, 0.11459784, 0.03594993, 0.25, 0.02203078, 0.05390004, 0.06215049, 0.09743458, 0.25])
    # balance_priors(priors, random=True)

    # print(np.sum(priors))
    # pd.DataFrame({"threshold": thresholds, "priors": priors}).round(3).T

    partition_optimal = find_partitions_optimal(X, thresholds, priors, threshold_true, c)
    acc_loss_optimal = evaluate_system(X, partition_optimal, thresholds, priors, threshold_true, c)


    # print("Greedy")
    # print("------")
    # print(f"Partition: {sorted(partition_greedy)}")
    # print(f"Acc Loss : {acc_loss_greedy:.4f}")
    # print()
    print("Optimal")
    print("-------")
    print(f"Partition: {sorted(partition_optimal)}")
    print(f"Acc Loss : {acc_loss_optimal:.4f}")