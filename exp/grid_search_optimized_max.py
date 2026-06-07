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
def generate_prior_grid(n_components, n_balls=100, p1=0.02, p2=0.02):
    """Fix the two near-t* classifiers (indices 1 and 2) to small constants and
    search the remaining mass over p0, p3, ..., p_{n-1}.

    Layout of each prior vector: [p0, p1, p2, p3, ..., p_{n-1}]
      - p1, p2          : fixed small constants (not searched)
      - p0, p3..p_{n-1} : searched over the simplex of mass (1 - p1 - p2)

    Every searched component is strictly positive: each gets at least one ball,
    then the remaining (n_balls - n_search) balls are distributed freely. So the
    minimum searched prior is step = (1 - p1 - p2) / n_balls.

    Grid size is C(n_balls - 1, n_search - 1) with n_search = n - 2.
    """
    assert n_components >= 3, "need at least 3 classifiers"

    n_search = n_components - 2            # p0 plus p3 .. p_{n-1}
    search_total = 1.0 - p1 - p2
    assert search_total > 0, "p1 + p2 must be < 1"
    assert n_balls >= n_search, "need n_balls >= n_components - 2 for all-positive priors"

    step = search_total / n_balls
    extra = n_balls - n_search            # balls left after seeding each component with 1
    grids = []
    for combo in itertools.combinations_with_replacement(range(n_search), extra):
        counts = np.bincount(combo, minlength=n_search) + 1   # each component >= 1 ball
        masses = counts * step
        priors = np.empty(n_components)
        priors[0] = masses[0]      # p0   (searched, > 0)
        priors[1] = p1             # fixed small
        priors[2] = p2             # fixed small
        priors[3:] = masses[1:]    # p3 .. p_{n-1}  (searched, > 0)
        grids.append(priors)
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
# Streaming aggregation
# ----------------------------------------------------------------------------
# At n=7 the prior grid is ~18M points; materializing one row per prior is not
# viable. Instead each worker keeps an O(top_k + n_bins) summary:
#   - counts of which path produced the optimum (agg / div / brute),
#   - the top_k worst priors by each ratio metric (full payload retained so
#     you can re-run / locally refine those),
#   - per-baseline-loss-bin count / sum / max of min(r_agg, r_div), which is
#     the ratio-vs-baseline-loss curve.
# merge_aggregates() combines the per-worker summaries.
# ============================================================================
def make_aggregator(top_k=500, n_bins=50, loss_max=1.0):
    return {
        "n": 0,
        "opt_source": {"agg": 0, "div": 0, "brute": 0},
        "top_agg": [], "top_div": [], "top_min": [],   # min-heaps of (r, ctr, payload)
        "top_k": top_k,
        "n_bins": n_bins,
        "loss_max": loss_max,
        "bin_count": np.zeros(n_bins),
        "bin_sum_min": np.zeros(n_bins),
        "bin_max_min": np.zeros(n_bins),
        "_ctr": 0,
    }


def _push_topk(heap, k, value, payload, ctr):
    if value is None or np.isnan(value):
        return
    if len(heap) < k:
        heapq.heappush(heap, (value, ctr, payload))
    elif value > heap[0][0]:
        heapq.heapreplace(heap, (value, ctr, payload))


def _update_aggregator(agg, priors, loss_base, loss_opt,
                       loss_greedy_agg, loss_greedy_div,
                       partition_opt, partition_greedy_agg, partition_greedy_div,
                       opt_source):
    agg["n"] += 1
    agg["opt_source"][opt_source] += 1

    r_agg = approximation_ratio(loss_opt, loss_greedy_agg, "m")
    r_div = approximation_ratio(loss_opt, loss_greedy_div, "m")
    finite = [r for r in (r_agg, r_div) if r is not None and not np.isnan(r)]
    r_min = min(finite) if finite else np.nan

    payload = {
        "priors": priors.astype(np.float32),
        "loss_base": loss_base, "loss_opt": loss_opt,
        "loss_greedy_agg": loss_greedy_agg, "loss_greedy_div": loss_greedy_div,
        "partition_opt": partition_opt,
        "partition_greedy_agg": partition_greedy_agg,
        "partition_greedy_div": partition_greedy_div,
        "r_mult_agg": r_agg, "r_mult_div": r_div,
    }
    ctr = agg["_ctr"]; agg["_ctr"] += 1
    _push_topk(agg["top_agg"], agg["top_k"], r_agg, payload, ctr)
    _push_topk(agg["top_div"], agg["top_k"], r_div, payload, ctr)
    _push_topk(agg["top_min"], agg["top_k"], r_min, payload, ctr)

    b = min(int(loss_base / agg["loss_max"] * agg["n_bins"]), agg["n_bins"] - 1)
    b = max(b, 0)
    agg["bin_count"][b] += 1
    if not np.isnan(r_min):
        agg["bin_sum_min"][b] += r_min
        agg["bin_max_min"][b] = max(agg["bin_max_min"][b], r_min)


def merge_aggregates(aggs):
    aggs = [a for a in aggs if a is not None]
    out = make_aggregator(aggs[0]["top_k"], aggs[0]["n_bins"], aggs[0]["loss_max"])
    ctr = 0
    for a in aggs:
        out["n"] += a["n"]
        for k in out["opt_source"]:
            out["opt_source"][k] += a["opt_source"][k]
        out["bin_count"] += a["bin_count"]
        out["bin_sum_min"] += a["bin_sum_min"]
        out["bin_max_min"] = np.maximum(out["bin_max_min"], a["bin_max_min"])
        for name in ("top_agg", "top_div", "top_min"):
            for value, _, payload in a[name]:
                _push_topk(out[name], out["top_k"], value, payload, ctr)
                ctr += 1
    return out


def aggregator_to_frames(agg):
    """Turn a merged aggregator into (worst_cases_df, baseline_curve_df)."""
    worst = pd.DataFrame(sorted((p for _, _, p in agg["top_min"]),
                                key=lambda d: -(min(r for r in (d["r_mult_agg"], d["r_mult_div"])
                                                    if r is not None and not np.isnan(r))
                                                if any(r is not None and not np.isnan(r)
                                                       for r in (d["r_mult_agg"], d["r_mult_div"]))
                                                else -np.inf)))
    edges = np.linspace(0, agg["loss_max"], agg["n_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_min = np.where(agg["bin_count"] > 0, agg["bin_sum_min"] / agg["bin_count"], np.nan)
    curve = pd.DataFrame({
        "baseline_loss": centers,
        "count": agg["bin_count"],
        "mean_r_min": mean_min,
        "max_r_min": agg["bin_max_min"],
    })
    return worst, curve


# ============================================================================
# Parallel driver
# ============================================================================
def process_chunk(priors_chunk, X, X_eps, thresholds, threshold_true, c, n,
                  partition_base, eps=1e-9, validate=False, tol=1e-9,
                  collect="full", top_k=500, n_bins=50, loss_max=1.0):
    """Process one chunk of the prior grid in a single worker.

    A fresh memo is built per prior, so memory stays bounded while every
    redundant block evaluation within that prior is eliminated.

    Greedy runs first: if its output carries an optimality certificate we
    take it as the optimum and skip the brute-force search. Set validate=True
    to additionally brute-force the certified priors and assert the certificate
    holds -- run this once on a small grid to be safe.

    collect="full"      -> return a list of per-prior dicts (small grids only).
    collect="aggregate" -> return an O(top_k + n_bins) summary (use at n>=7).
    """
    rows = [] if collect == "full" else None
    agg = make_aggregator(top_k, n_bins, loss_max) if collect == "aggregate" else None

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

        if collect == "full":
            rows.append({
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
        else:
            _update_aggregator(agg, priors, loss_base, loss_opt,
                               loss_greedy_agg, loss_greedy_div,
                               partition_opt, partition_greedy_agg, partition_greedy_div,
                               opt_source)

    return rows if collect == "full" else agg


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

    N_THRESHOLDS = 6
    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)

    # NOTE: n_balls drives the grid size hard. With p1, p2 fixed and all searched
    # priors strictly positive, the grid is C(n_balls - 1, n_search - 1) priors,
    # where n_search = N_THRESHOLDS - 2. At N_THRESHOLDS=7 (n_search = 5):
    #   n_balls=100 -> ~3.76M    n_balls=50 -> ~212k    n_balls=30 -> ~24k
    n_balls = 92
    p1_fixed = 0.04              # near-t* classifiers held small, not searched
    p2_fixed = 0.04
    # thresholds = np.linspace(0.0, 1.0, N_THRESHOLDS)
    if N_THRESHOLDS == 6:
        thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif N_THRESHOLDS == 7:
        thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.7333, 0.8666, 1.0])
    prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls, p1=p1_fixed, p2=p2_fixed)
    threshold_true = 0.2
    c = 0.75
    X_eps = np.arange(-1 / c, 0, 1e-4).round(4)
    partition_base = [[i] for i in range(N_THRESHOLDS)]

    # ---- run config ---------------------------------------------------------
    N_JOBS = -1                      # -1 = all cores
    VALIDATE = False                 # set True on a SMALL grid to check certs
    COLLECT = "full"            # "aggregate" for n>=7, "full" for small grids
    TOP_K = 1000                     # worst cases retained (full payload)
    N_BINS = 50                      # baseline-loss bins for the ratio curve

    n_workers = joblib.cpu_count() if N_JOBS == -1 else N_JOBS
    n_chunks = min(len(prior_grid), max(n_workers * 8, n_workers))
    chunks = chunkify(prior_grid, n_chunks)

    print(f"{len(prior_grid)} priors over {len(chunks)} chunks on {n_workers} workers "
          f"(collect={COLLECT})")

    with tqdm_joblib(tqdm.tqdm(total=len(chunks), desc="chunks")):
        results_nested = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk)(
                chunk, X, X_eps, thresholds, threshold_true, c,
                N_THRESHOLDS, partition_base,
                validate=VALIDATE, collect=COLLECT, top_k=TOP_K, n_bins=N_BINS,
            )
            for chunk in chunks
        )

    if COLLECT == "full":
        results = [r for sub in results_nested for r in sub]
        df_results = pd.DataFrame(results)
        print(df_results.shape)
        print(df_results["opt_source"].value_counts())
        filepath = f"grid_search_n{N_THRESHOLDS}_refined_p1_{p1_fixed}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(df_results, f)
        print(f"Saved to {filepath}")
    else:
        agg = merge_aggregates(results_nested)
        worst_df, curve_df = aggregator_to_frames(agg)
        print(f"priors processed: {agg['n']}")
        print(f"opt source counts: {agg['opt_source']}")
        if len(worst_df):
            print(f"top worst-case r_min: {worst_df.iloc[0]['r_mult_agg']:.4g} (agg) / "
                  f"{worst_df.iloc[0]['r_mult_div']:.4g} (div)")
        filepath = f"grid_search_n{N_THRESHOLDS}_agg.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({"aggregator": agg, "worst": worst_df, "curve": curve_df,
                         "meta": {"c": c, "threshold_true": threshold_true,
                                  "thresholds": thresholds, "n_balls": n_balls,
                                  "p1_fixed": p1_fixed, "p2_fixed": p2_fixed}}, f)
        print(f"Saved to {filepath}")