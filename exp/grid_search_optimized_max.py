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
from collections import defaultdict
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


def filter_partitions(partitions):
    final = []
    for partition in partitions:
        if len(partition) == 2:
            if len(partition[0]) == 2 or len(partition[0]) == 4:
                final.append(partition)
    return final


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
    partitions = set_partitions(list(range(n)))
    partitions = filter_partitions(partitions)
    for partition in partitions:
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
def generate_prior_grid(n_components, n_balls=50):
    """Search the full simplex with every prior strictly positive.

    Each of the n_components gets at least one ball (=> prior >= 1/n_balls), then
    the remaining (n_balls - n_components) balls are distributed freely.
    Grid size is C(n_balls - 1, n_components - 1).
    """
    step = 1.0 / n_balls
    grids = []
    for combo in itertools.combinations_with_replacement(range(n_components), n_balls - n_components):
        counts = np.bincount(combo, minlength=n_components) + 1
        grids.append((counts * step).round(4))
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


def process_combo_chunk(prior_chunk, c, threshold_true, X, thresholds, n,
                        partition_base, validate=False, collect="aggregate",
                        top_k=500, n_bins=50, loss_max=1.0):
    """One task = (prior_chunk, c, threshold_true). Builds X_eps from c, runs the
    chunk, and tags the result with (c, threshold_true) so results can be grouped
    back per sweep cell."""
    X_eps = np.arange(-1.0 / c, 0, 1e-4).round(4)
    result = process_chunk(prior_chunk, X, X_eps, thresholds, threshold_true, c, n,
                           partition_base, validate=validate, collect=collect,
                           top_k=top_k, n_bins=n_bins, loss_max=loss_max)
    return (c, threshold_true, result)


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

    N_THRESHOLDS = 7
    n_balls = 25
    X = np.arange(0., 1. + 1e-4, 1e-4).round(4)
    # thresholds = np.linspace(0.0, 1.0, N_THRESHOLDS)
    thresholds = np.array([0.0, 0.2, 0.4, 0.6, 0.7333, 0.8666, 1.0])
    prior_grid = generate_prior_grid(N_THRESHOLDS, n_balls=n_balls)   # full positive simplex
    partition_base = [[i] for i in range(N_THRESHOLDS)]

    # ---- sweep axes ---------------------------------------------------------
    c_values = [1.0]
    tt_values = [0.15]
    combos = [(c, tt) for c in c_values for tt in tt_values]

    # ---- run config ---------------------------------------------------------
    N_JOBS = -1                      # -1 = all cores
    VALIDATE = False                 # set True to brute-check greedy certificates
    TOP_K = 500                      # worst priors retained per (c, t*) cell
    N_BINS = 50                      # baseline-loss bins for the ratio curve
    PRIOR_CHUNK_SIZE = 20000         # priors per task

    # Chunk all three axes: tasks = (prior_chunk x c x t*). Each task aggregates
    # its chunk; aggregators are later grouped by (c, t*) and merged so every
    # sweep cell keeps its own worst-cases + curve.
    n_prior_chunks = max(1, len(prior_grid) // PRIOR_CHUNK_SIZE)
    prior_chunks = chunkify(prior_grid, n_prior_chunks)
    tasks = [(pc, c, tt) for (c, tt) in combos for pc in prior_chunks]

    n_workers = joblib.cpu_count() if N_JOBS == -1 else N_JOBS
    print(f"{len(prior_grid):_} priors x {len(combos)} (c, t*) combos "
          f"= {len(prior_grid) * len(combos):_} evaluations; "
          f"{len(tasks)} tasks ({len(prior_chunks)} prior-chunks/cell) on {n_workers} workers")

    with tqdm_joblib(tqdm.tqdm(total=len(tasks), desc="tasks")):
        results_flat = Parallel(n_jobs=N_JOBS)(
            delayed(process_combo_chunk)(
                pc, c, tt, X, thresholds, N_THRESHOLDS, partition_base,
                validate=VALIDATE, collect="aggregate",
                top_k=TOP_K, n_bins=N_BINS,
            )
            for (pc, c, tt) in tasks
        )

    # ---- group the per-chunk aggregators back into one per (c, t*) ----------
    groups = defaultdict(list)
    for c, tt, agg in results_flat:
        groups[(c, tt)].append(agg)

    per_combo = {}
    summary_rows = []
    for (c, tt), aggs in groups.items():
        merged = merge_aggregates(aggs)
        worst_df, curve_df = aggregator_to_frames(merged)
        per_combo[(c, tt)] = {"agg": merged, "worst": worst_df, "curve": curve_df}

        max_r_min = np.nan
        if len(worst_df):
            top = worst_df.iloc[0]
            finite = [r for r in (top["r_mult_agg"], top["r_mult_div"]) if pd.notna(r)]
            max_r_min = min(finite) if finite else np.nan
        summary_rows.append({"c": c, "threshold_true": tt,
                             "max_r_min": max_r_min, "n": merged["n"],
                             "opt_brute": merged["opt_source"]["brute"]})

    summary = pd.DataFrame(summary_rows).sort_values(["c", "threshold_true"])
    pivot = summary.pivot(index="c", columns="threshold_true", values="max_r_min")
    print("\nmax r_min by (c, threshold_true):")
    print(pivot.round(3))

    overall = summary.loc[summary["max_r_min"].idxmax()] if summary["max_r_min"].notna().any() else None
    if overall is not None:
        print(f"\npeak ratio {overall['max_r_min']:.4g} at "
              f"c={overall['c']}, t*={overall['threshold_true']}")

    filepath = f"grid_search_n{N_THRESHOLDS}_c_tt_sweep_heuristic_exact.pkl"
    with open(filepath, "wb") as f:
        pickle.dump({"per_combo": per_combo, "summary": summary, "pivot": pivot,
                     "meta": {"thresholds": thresholds, "n_balls": n_balls,
                              "c_values": c_values, "tt_values": tt_values,
                              "top_k": TOP_K, "n_bins": N_BINS}}, f)
    print(f"Saved to {filepath}")