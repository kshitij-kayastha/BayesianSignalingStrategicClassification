import cvxpy as cp
import itertools
import numpy as np
import pandas as pd
import time
import tqdm
import warnings

warnings.filterwarnings("ignore")

def validate(priors, thresholds, tt, c):
    p = np.asarray(priors, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    if p.shape != t.shape or p.ndim != 1:
        raise ValueError("priors and thresholds must be 1-D of equal length")
    if np.any(p < 0):
        raise ValueError("priors must be nonnegative")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError(f"priors must sum 1, got {p.sum()}")
    if c <= 0:
        raise ValueError("c must be positive")
    
    return p, t, float(tt), float(c)

def accept_matrix(thresholds):
    t = np.asarray(thresholds, dtype=float)
    return (t[None, :] >= t[:, None]).astype(float)

def build_grid(m, lo=0.0, hi=1.0):
    X = np.linspace(lo, hi, m)
    D = np.full(m, 1.0/m)
    return X, D

def build_constants(priors, thresholds, tt, c, m, eps=1e-6):
    priors, thresholds, tt, c = validate(priors, thresholds, tt, c)
    n = len(thresholds)
    X, D = build_grid(m)

    A = np.empty((m, n+1))
    A[:, 0] = X
    A[:, 1:] = thresholds[None, :]

    cost = c * np.abs(A - X[:, None])
    Hx = (A[:, None, :] >= thresholds[None, :, None]).astype(float)
    U = priors[None, :, None] * (Hx - (1.0 + eps) * cost[:, None, :])
    f = (X >= tt).astype(float)
    L = np.where(f[:, None, None] == 0.0, Hx, 1.0 - Hx)
    M = 1.0 + (1.0 + eps) * cost.max(axis=1)

    valid = A >= X[:, None] - 1e-12
    for xi in range(m):
        seen = set()
        for a in range(n+1):
            key = round(float(A[xi, a]), 12)
            if key in seen:
                valid[xi, a] = False
            else:
                seen.add(key)
    return dict(p=priors, t=thresholds, tt=tt, c=c, n=n, m=m, X=X, D=D, A=A, cost=cost, Hx=Hx, U=U, L=L, M=M, eps=eps, valid=valid)

def build_variables(K):
    n, m = K["n"], K["m"]

    xv = cp.Variable((n, n), boolean=True, name="x")
    yv = cp.Variable((m*n, n+1), boolean=True, name="y")

    cons = [
        cp.sum(xv, axis=1) == 1,
        cp.sum(yv, axis=1) == 1,
    ]
    return xv, yv, cons

def row(x_idx, j, n):
    return x_idx * n + j

def assignment_to_partition(assign):
    bins = {}
    for i, j in enumerate(assign):
        bins.setdefault(j, []).append(i)
    return frozenset(frozenset(v) for v in bins.values())

def bell(n):
    r = [1]
    for _ in range(n):
        new = [r[-1]]
        for v in r:
            new.append(new[-1] + v)
        r = new
    return r[0]

def add_best_response(K, xv, yv):
    n, m, U, M, valid = K["n"], K["m"], K["U"], K["M"], K["valid"]
    ones = np.ones((1, n+1))
    cons = []

    for x_idx in range(m):
        util = xv.T @ U[x_idx]
        y_x = yv[row(x_idx, 0, n): row(x_idx, 0, n)+n, :]

        for a in range(n+1):
            if not valid[x_idx, a]:
                cons.append(y_x[:, a] == 0)
                continue
            u_a = cp.reshape(util[:, a], (n,1), order="C")
            y_a = cp.reshape(y_x[:, a], (n,1), order="C")
            cons.append((u_a + M[x_idx] * (1 - y_a)) @ ones >= util)
    return cons

def chosen_landing(K, yv_value):
    n, m = K["n"], K["m"]
    Y = np.asarray(yv_value).reshape(m, n, n+1)
    a = Y.argmax(axis=2)
    return K["A"][np.arange(m)[:, None], a]

def brute_force_landing(K, assign):
    n, m = K["n"], K["m"]
    out = np.zeros((m, n))
    for j in range(n):
        B = (assign == j).astype(float)
        if B.sum() == 0:
            out[:, j] = np.nan
            continue
        score = (K["U"] + B[None, :, None]).sum(axis=1)
        score = np.where(K["valid"], score, -np.inf)
        out[:, j] = K["A"][np.arange(m), score.argmax(axis=1)]
    return out

def add_objective(K, xv, yv):
    n, m , L, D, p = K["n"], K["m"], K["L"], K["D"], K["p"]

    v = cp.Variable((m * n, n), nonneg=True, name="v")
    cons = []
    for x_idx in range(m):
        sl = slice(x_idx * n, (x_idx + 1) * n)
        g = yv[sl, :] @ L[x_idx].T
        cons.append(v[sl, :] >= g + xv.T - 1)

    W = np.repeat(D, n)[:, None] * p[None, :]
    return cp.sum(cp.multiply(W, v)), cons

def evaluate_partition(K, assign):
    n, m, U, L, D, p, valid = K["n"], K["m"], K["U"], K["L"], K["D"], K["p"], K["valid"]

    total = 0.0
    for j in range(n):
        members = np.flatnonzero(assign == j)
        if members.size == 0:
            continue
        B = np.zeros(n)
        B[members] = 1.0
        score = (U * B[None, :, None]).sum(axis=1)
        a_star = np.where(valid, score, -np.inf).argmax(axis=1)
        err = L[np.arange(m)[:, None], members[None, :], a_star[:, None]]
        total += float(D @ (err @ p[members]))
    return total

def add_symmetry(K, xv):
    n = K["n"]
    cons = [xv[i, j] == 0 for i in range(n) for j in range(i+1, n)]
    for i in range(1, n):
        for j in range(1, i+1):
            cons.append(xv[i,j] <= cp.sum(xv[:i, j-1]))
    return cons

def pin_empty_bins(K, xv, yv):
    n = K["n"]
    occ = cp.sum(xv, axis=0)
    return [yv[x_idx * n:(x_idx + 1) * n, 0] >= 1 - occ for x_idx in range(K["m"])]

def solve(priors, thresholds, tt, c, m=51, eps=1e-6, prune=True, pin_empty=True, feas_tol=1e-9, check_tol=1e-9, time_limit=None):
    K = build_constants(priors, thresholds, tt, c, m, eps)
    xv, yv, cons = build_variables(K)
    cons = cons + add_best_response(K, xv, yv)
    if prune:
        cons = cons + add_symmetry(K, xv)
    if pin_empty:
        cons = cons + pin_empty_bins(K, xv, yv)
    obj, obj_cons = add_objective(K, xv, yv)

    prob = cp.Problem(cp.Minimize(obj), cons + obj_cons)
    kw = dict(primal_feasibility_tolerance=feas_tol, mip_feasibility_tolerance=feas_tol)
    if time_limit is not None:
        kw["time_limit"] = time_limit
    
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.HIGHS, **kw)
    elapsed = time.perf_counter() - t0

    assign = np.asarray(xv.value).argmax(axis=1)
    loss = evaluate_partition(K, assign)
    return dict(
        status=prob.status,
        partition=sorted(sorted(b) for b in assignment_to_partition(assign)),
        loss=loss,
        solver_objective=float(prob.value),
        certified=bool(loss - float(prob.value) < check_tol),
        seconds=elapsed,
        K=K
    )

def brute_force(K):
    t0 = time.perf_counter()
    n = K["n"]
    best = (np.inf, None)
    for assign in itertools.product(range(n), repeat=n):
        val = evaluate_partition(K, np.array(assign))
        if val < best[0] - 1e-12:
            best = (val, np.array(assign))
    return best, time.perf_counter() - t0