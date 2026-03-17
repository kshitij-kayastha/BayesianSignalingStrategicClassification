import collections
import heapq
import itertools
import numpy as np
from copy import deepcopy
from functools import lru_cache
from .subroutine import evaluate_partition
from .util import display_queue, set_partitions


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

    for partitions in partitions_set:

        acc_loss = 0.0
        for partition in partitions:
            partition_tuple = tuple(sorted(partition))  # must be hashable
            acc_loss += evaluate_partition_cached(partition_tuple)

        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = deepcopy(partitions)

    return best_partition

def find_partitions_greedy(X, thresholds, priors, threshold_true, c, display=False):
    P = {}
    next_id = 0
    partitions = [[i] for i in range(len(priors))]
    for block in partitions:
        P[next_id] = list(block)
        next_id += 1

    Q = collections.deque(itertools.combinations(P.keys(), 2))

    while Q:
        if display:
            display_queue(Q, P)
        a_id, b_id = Q.popleft()
        if a_id not in P.keys() or b_id not in P.keys():
            continue

        a = P[a_id]
        b = P[b_id]

        acc_loss_a = evaluate_partition(X, a, thresholds, priors, threshold_true, c)
        acc_loss_b = evaluate_partition(X, b, thresholds, priors, threshold_true, c)
        lhs = acc_loss_a * np.sum(priors[a]) + acc_loss_b * np.sum(priors[b])

        ab = sorted(a + b)
        acc_loss_ab = evaluate_partition(X, ab, thresholds, priors, threshold_true, c)
        rhs = acc_loss_ab * np.sum(priors[ab])

        if lhs - rhs > -1e-6:
            del P[a_id]
            del P[b_id]

            Q = collections.deque(
                (x, y)
                for (x, y) in Q
                if x not in {a_id, b_id} and y not in {a_id, b_id}
            )

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for c_id in P.keys():
                if c_id != new_id:
                    Q.append((new_id, c_id))

            # Q = collections.deque(sorted(Q, key=lambda x: P[x[0]]))
    return list(P.values())


def find_partitions_greedy_best(X, thresholds, priors, threshold_true, c):
    P = {}
    next_id = 0
    partitions = [[i] for i in range(len(priors))]
    for block in partitions:
        P[next_id] = block
        next_id += 1
    
    Q = collections.deque(itertools.combinations(P.keys(), 2))
    pq = []
    for a_id, b_id in Q:
        acc_loss_ab = evaluate_partition(X, sorted(P[a_id]+P[b_id]), thresholds, priors, threshold_true, c)
        heapq.heappush(pq, (acc_loss_ab, (a_id, b_id)))

    while pq:
        acc_loss_ab, (a_id, b_id) = heapq.heappop(pq)
        if a_id not in P.keys() or b_id not in P.keys():
            continue

        a = P[a_id]
        b = P[b_id]

        acc_loss_a = evaluate_partition(X, a, thresholds, priors, threshold_true, c)
        acc_loss_b = evaluate_partition(X, b, thresholds, priors, threshold_true, c)
        lhs = acc_loss_a * np.sum(priors[a]) + acc_loss_b * np.sum(priors[b])

        ab = sorted(a + b)
        rhs = acc_loss_ab * np.sum(priors[ab])

        if lhs > rhs:
            del P[a_id]
            del P[b_id]

            pq2 = []
            for acc_loss, (x_id, y_id) in pq:
                if x_id not in {a_id, b_id} and y_id not in {a_id, b_id}:
                    acc_loss = evaluate_partition(X, sorted(P[x_id]+P[y_id]), thresholds, priors, threshold_true, c)
                    heapq.heappush(pq2, (acc_loss, (x_id,y_id)))
            pq = deepcopy(pq2)

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for p_id in P.keys():
                if p_id != new_id:
                    acc_loss = evaluate_partition(X, sorted(P[p_id]+P[new_id]), thresholds, priors, threshold_true, c)
                    heapq.heappush(pq, (acc_loss, (new_id, p_id)))
    return list(P.values())