import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tqdm
import pickle
import heapq
import itertools
import contextlib
import joblib
from joblib import Parallel, delayed

from src.subroutine import best_response_vectorized, accuracy_loss_vectorized


# ============================================================================
# Combinatorial helpers (unchanged)
# ============================================================================
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


# ============================================================================
# Original loss helpers (kept for compatibility with other utilities, e.g.
# construct_worst_case / find_worst_case). The grid search itself no longer
# calls these directly -- it goes through the memoized layer below.
# ============================================================================
def compute_partition_loss(X, partition, thresholds, priors, threshold_true, c):
    thresholds_p = thresholds[partition]
    priors_p = priors[partition]
    X_p = best_response_vectorized(X, thresholds_p, priors_p, c)
    return accuracy_loss_vectorized(X, X_p, thresholds_p, priors_p, threshold_true)


def compute_system_loss(X, partitions, thresholds, priors, threshold_true, c):
    acc_loss = 0.0
    for partition in partitions:
        acc_loss_p = compute_partition_loss(X, partition, thresholds, priors, threshold_true, c)
        acc_loss += acc_loss_p * np.sum(priors[partition])
    return acc_loss


def approximation_ratio(loss_optimal, loss_greedy, rtype="m"):
    if rtype in ["a", "add", "additive"]:
        return loss_greedy - loss_optimal
    elif rtype in ["m", "mult", "multiplicative"]:
        if loss_optimal == 0:
            return np.nan
        return loss_greedy / loss_optimal


# ============================================================================
# Per-prior memoized block-loss cache
# ----------------------------------------------------------------------------
# All three searches (agglomerative / divisive / optimal) repeatedly evaluate
# the loss of the SAME block for a given prior. We memoize on (data_tag, block)
# so each unique block is fed through best_response_vectorized at most once per
# prior. The cache is local to a single prior, so it stays tiny (<= 2*(2^n - 1)
# entries) and there is no cross-process state to manage.
#
# tag = 0 -> evaluate against X      (used by all three searches)
# tag = 1 -> evaluate against X_eps  (used only by the greedy tie-breaker)
# `block` is always a sorted tuple of indices, so keys are canonical.
# ============================================================================
def make_block_loss(X, X_eps, thresholds, priors, threshold_true, c):
    loss_cache = {}
    psum_cache = {}

    def psum(block):
        s = psum_cache.get(block)
        if s is None:
            s = float(priors[list(block)].sum())
            psum_cache[block] = s
        return s

    def loss(tag, block):
        key = (tag, block)
        v = loss_cache.get(key)
        if v is None:
            data = X if tag == 0 else X_eps
            idx = list(block)
            thr = thresholds[idx]
            pri = priors[idx]
            data_br = best_response_vectorized(data, thr, pri, c)
            v = accuracy_loss_vectorized(data, data_br, thr, pri, threshold_true)
            loss_cache[key] = v
        return v

    return loss, psum


# ============================================================================
# Cached searches. Logic is identical to the originals; only the loss
# computation is routed through the memo. (Exception: optimal uses proper
# system loss -- see note in the accompanying message.)
# ============================================================================
def optimal_cached(loss, psum, n):
    best_partition, best_loss = None, np.inf
    for partition in set_partitions(list(range(n))):
        acc_loss = 0.0
        for block in partition:
            b = tuple(sorted(block))
            acc_loss += loss(0, b) * psum(b)
        if acc_loss < best_loss:
            best_loss = acc_loss
            best_partition = partition
    return best_partition


def agglomerative_cached(loss, psum, n, eps=1e-9):
    def wcost(block):
        b = tuple(sorted(block))
        return (loss(0, b) + loss(1, b) * eps) * psum(b)

    P = {i: [i] for i in range(n)}
    next_id = n

    pq = []
    for a_id, b_id in itertools.combinations(list(P.keys()), 2):
        a, b = P[a_id], P[b_id]
        ab = sorted(a + b)
        gain = -(wcost(a) + wcost(b) - wcost(ab))
        heapq.heappush(pq, (gain, (a_id, b_id)))

    while pq:
        gain, (a_id, b_id) = heapq.heappop(pq)
        if a_id not in P or b_id not in P:
            continue
        a, b = P[a_id], P[b_id]
        ab = sorted(a + b)

        lhs = wcost(a) + wcost(b)
        rhs = wcost(ab)

        if lhs - rhs > -1e-9:
            del P[a_id]
            del P[b_id]
            pq = [(g, (x, y)) for g, (x, y) in pq
                  if x not in {a_id, b_id} and y not in {a_id, b_id}]
            heapq.heapify(pq)

            new_id = next_id
            next_id += 1
            P[new_id] = ab

            for p_id in list(P.keys()):
                if p_id == new_id:
                    continue
                p = P[p_id]
                merged = sorted(ab + p)
                gain = -(wcost(p) + wcost(ab) - wcost(merged))
                heapq.heappush(pq, (gain, (new_id, p_id)))
    return list(P.values())


def divisive_cached(loss, psum, n, eps=1e-9):
    def bcost(block):
        b = tuple(sorted(block))
        return loss(0, b) + loss(1, b) * eps

    partition_0 = [list(range(n))]
    acc_loss_0 = bcost(partition_0[0])
    pq = [(acc_loss_0, partition_0)]

    while pq:
        acc_loss_merged, partition_merged = heapq.heappop(pq)
        if len(partition_merged[0]) == 1:
            return partition_merged
        partitions = split_partition(partition_merged)
        pq = []
        for partition in partitions:
            acc_loss_split = 0.0
            for block in partition:
                acc_loss_split += bcost(block) * psum(tuple(sorted(block)))
            gain = acc_loss_merged - acc_loss_split
            if gain > 0:
                heapq.heappush(pq, (acc_loss_split, partition))
        if not pq:
            return partition_merged


def system_loss_cached(partition, loss, psum):
    total = 0.0
    for block in partition:
        b = tuple(sorted(block))
        total += loss(0, b) * psum(b)
    return total


# ============================================================================
# Grid generators (unchanged)
# ============================================================================
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


# ============================================================================
# Optimality certificates
# ----------------------------------------------------------------------------
# When a greedy search lands on one of these structures, the result is provably
# optimal, so we can skip the brute-force enumeration for that prior entirely.
# Checked by block-size signature, so it is index-agnostic (any pair / any
# singleton qualifies).
# ============================================================================
def _block_sizes(partition):
    return sorted(len(b) for b in partition)


def agg_certifies_optimal(partition, n):
    """Agglomerative output known to be optimal:
       - all singletons             -> sizes [1, ..., 1]
       - one pair + rest singletons -> sizes [1, ..., 1, 2]
    """
    sizes = _block_sizes(partition)
    return sizes == [1] * n or sizes == [1] * (n - 2) + [2]


def div_certifies_optimal(partition, n):
    """Divisive output known to be optimal:
       - grand partition            -> sizes [n]
       - one singleton + rest        -> sizes [1, n - 1]
    """
    sizes = _block_sizes(partition)
    return sizes == [n] or sizes == [1, n - 1]


# ============================================================================
# Parallel driver
# ============================================================================
def process_chunk(priors_chunk, X, X_eps, thresholds, threshold_true, c, n,
                  partition_base, eps=1e-9, validate=False, tol=1e-9):
    """Process one chunk of the prior grid in a single worker.

    A fresh memo is built per prior, so memory stays bounded while every
    redundant block evaluation within that prior is eliminated.

    Greedy runs first: if its output carries an optimality certificate we
    take it as the optimum and skip the brute-force search (and therefore the
    63 X-block evaluations brute force would otherwise force). Set
    validate=True to additionally run brute force on the certified priors and
    assert the certificate holds -- run this once on a small grid to be safe.
    """
    out = []
    for priors in priors_chunk:
        priors = np.asarray(priors, dtype=float)
        loss, psum = make_block_loss(X, X_eps, thresholds, priors, threshold_true, c)

        # Greedy first -- cheap, and a certified structure short-circuits the
        # expensive enumeration below.
        partition_greedy_agg = agglomerative_cached(loss, psum, n, eps)
        partition_greedy_div = divisive_cached(loss, psum, n, eps)
        loss_greedy_agg = system_loss_cached(partition_greedy_agg, loss, psum)
        loss_greedy_div = system_loss_cached(partition_greedy_div, loss, psum)

        if agg_certifies_optimal(partition_greedy_agg, n):
            partition_opt, loss_opt, opt_source = partition_greedy_agg, loss_greedy_agg, "agg"
        elif div_certifies_optimal(partition_greedy_div, n):
            partition_opt, loss_opt, opt_source = partition_greedy_div, loss_greedy_div, "div"
        else:
            partition_opt = optimal_cached(loss, psum, n)
            loss_opt = system_loss_cached(partition_opt, loss, psum)
            opt_source = "brute"

        if validate and opt_source != "brute":
            partition_brute = optimal_cached(loss, psum, n)
            loss_brute = system_loss_cached(partition_brute, loss, psum)
            if not np.isclose(loss_opt, loss_brute, atol=tol, rtol=0.0):
                raise AssertionError(
                    f"certificate '{opt_source}' gave loss {loss_opt:.12g} but "
                    f"brute-force optimum is {loss_brute:.12g} for "
                    f"priors={priors.tolist()}"
                )

        loss_base = system_loss_cached(partition_base, loss, psum)

        out.append({
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
            "opt_source": opt_source,
            "r_mult_agg": approximation_ratio(loss_opt, loss_greedy_agg, "m"),
            "r_mult_div": approximation_ratio(loss_opt, loss_greedy_div, "m"),
            "r_add_agg": approximation_ratio(loss_opt, loss_greedy_agg, "a"),
            "r_add_div": approximation_ratio(loss_opt, loss_greedy_div, "a"),
        })
    return out


def chunkify(seq, n_chunks):
    n_chunks = max(1, min(n_chunks, len(seq)))
    k, m = divmod(len(seq), n_chunks)
    return [seq[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Let a tqdm bar advance as joblib tasks complete."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


if __name__ == "__main__":
    np.random.seed(0)

    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)
    N_THRESHOLDS = 6

    n_balls = 100
    n_balls_eps = 100
    thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls, eps=0.04, n_balls_eps=n_balls_eps)
    threshold_true = 0.2
    c = 0.75
    X_eps = np.arange(-1 / c, 0, 1e-4).round(4)
    partition_base = [[i] for i in range(N_THRESHOLDS)]

    # ---- parallelism config -------------------------------------------------
    N_JOBS = -1                      # -1 = all cores
    VALIDATE = False                 # set True on a SMALL grid to check certs
    n_workers = joblib.cpu_count()-2 if N_JOBS == -1 else N_JOBS
    # more chunks than workers => better load balance + a smoother progress bar.
    # arrays (X, X_eps, thresholds) are shipped once per chunk, not per prior.
    n_chunks = min(len(prior_grid), max(n_workers * 8, n_workers))
    chunks = chunkify(prior_grid, n_chunks)

    print(f"{len(prior_grid)} priors over {len(chunks)} chunks on {n_workers} workers")

    with tqdm_joblib(tqdm.tqdm(total=len(chunks), desc="chunks")):
        results_nested = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk)(
                chunk, X, X_eps, thresholds, threshold_true, c,
                N_THRESHOLDS, partition_base, validate=VALIDATE
            )
            for chunk in chunks
        )
    results = [r for sub in results_nested for r in sub]

    df_results = pd.DataFrame(results)
    print(df_results.shape)
    print(df_results["opt_source"].value_counts())

    filepath = f"grid_search_n{N_THRESHOLDS}_refined2.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(df_results, f)
    print(f"Saved to {filepath}")