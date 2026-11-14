"""Independent checks of US 12,363,024 B2 (Switching optimization).

Problem (patent paras [0080]-[0102], [0104]-[0129]):
  types s=1..m with predicted volumes T_s, networks k=1..n with unit costs c[s,k],
  all-units discount d_k activated (f_k = 1) only if routed volume >= V_k.
  True monetary cost: sum_k (1 - d_k f_k) * sum_s T_s c[s,k] x[s,k].

Checks
  A. Literal QUBO energy of [0099] scales the discount by the penalty P.
  B. With x fixed, the binary ("quantum") sub-problem of claim 12(b) is separable
     and solved in closed form f_k = 1[vol_k >= V_k]; compare with 2^n enumeration.
  C. Claim-12 alternation with *exact* sub-solvers vs. the global MILP optimum.
  D. Exact classical MILP runtime at production-like scale.
  E. Size of the literal QUBO vs. annealer clique limits.
"""
import itertools
import json
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

RNG = np.random.default_rng(20261101)
OUT = {}


def make_instance(m, n, rng, tight=True):
    T = rng.integers(1_000, 200_000, size=m).astype(float)
    base = rng.uniform(0.05, 0.20, size=n)
    c = base[None, :] * rng.uniform(0.85, 1.25, size=(m, n))
    d = rng.uniform(0.05, 0.30, size=n)
    share = rng.uniform(0.15, 0.6, size=n) if tight else rng.uniform(0.05, 0.3, size=n)
    V = share * T.sum()
    return T, c, d, V


def true_cost(x, f, T, c, d):
    C = (T[:, None] * c * x).sum(axis=0)
    return float(((1 - d * f) * C).sum())


def vol(x, T):
    return (T[:, None] * x).sum(axis=0)


# ---------------------------------------------------------------- MILP (global)
def solve_milp(T, c, d, V, integral_x=True, time_limit=600):
    m, n = c.shape
    nx = m * n
    # variables: x (m*n), w (m*n) with w = f*x, f (n)
    N = 2 * nx + n
    A = T[:, None] * c  # m x n
    cost = np.concatenate([A.ravel(), -(A * d[None, :]).ravel(), np.zeros(n)])
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for s in range(m):  # sum_k x = 1
        for k in range(n):
            rows.append(r); cols.append(s * n + k); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1
    for k in range(n):  # sum_s T_s x_sk - V_k f_k >= 0
        for s in range(m):
            rows.append(r); cols.append(s * n + k); vals.append(T[s])
        rows.append(r); cols.append(2 * nx + k); vals.append(-V[k])
        lb.append(0.0); ub.append(np.inf); r += 1
    for s in range(m):  # w <= x, w <= f
        for k in range(n):
            i = s * n + k
            rows += [r, r]; cols += [nx + i, i]; vals += [1.0, -1.0]
            lb.append(-np.inf); ub.append(0.0); r += 1
            rows += [r, r]; cols += [nx + i, 2 * nx + k]; vals += [1.0, -1.0]
            lb.append(-np.inf); ub.append(0.0); r += 1
    Acon = coo_matrix((vals, (rows, cols)), shape=(r, N)).tocsr()
    integrality = np.zeros(N)
    integrality[2 * nx:] = 1
    if integral_x:
        integrality[:nx] = 1
    t0 = time.perf_counter()
    res = milp(cost, constraints=LinearConstraint(Acon, lb, ub), integrality=integrality,
               bounds=Bounds(np.zeros(N), np.ones(N)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-9, "disp": False})
    dt = time.perf_counter() - t0
    x = res.x[:nx].reshape(m, n)
    f = np.round(res.x[2 * nx:])
    return x, f, true_cost(x, f, T, c, d), dt, res


# ---------------------------------------------------------------- claim 12 pieces
def f_step_bruteforce(x, T, c, d, V):
    n = c.shape[1]
    C = (T[:, None] * c * x).sum(axis=0)
    v = vol(x, T)
    best, bf = None, None
    for bits in itertools.product([0, 1], repeat=n):
        f = np.array(bits, float)
        if np.any(v + 1e-9 < V * f):
            continue
        val = float(((1 - d * f) * C).sum())
        if best is None or val < best - 1e-12:
            best, bf = val, f
    return bf


def f_step_closed_form(x, T, V):
    return (vol(x, T) + 1e-9 >= V).astype(float)


def x_step_lp(f, T, c, d, V):
    m, n = c.shape
    A = T[:, None] * c * (1 - d * f)[None, :]
    A_eq = csr_matrix((np.ones(m * n), (np.repeat(np.arange(m), n), np.arange(m * n))), shape=(m, m * n))
    b_eq = np.ones(m)
    active = [k for k in range(n) if f[k] > 0.5]
    if active:
        r_idx = np.repeat(np.arange(len(active)), m)
        c_idx = np.concatenate([np.arange(m) * n + k for k in active])
        vals = np.concatenate([-T for _ in active])
        A_ub = csr_matrix((vals, (r_idx, c_idx)), shape=(len(active), m * n))
        b_ub = np.array([-V[k] for k in active])
    else:
        A_ub, b_ub = None, None
    res = linprog(A.ravel(), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, 1), method="highs")
    if res.status != 0:
        return None
    return res.x.reshape(m, n)


def claim12(T, c, d, V, x0, max_iter=100):
    x = x0.copy()
    f = f_step_bruteforce(x, T, c, d, V)
    agree = 0
    total = 0
    for it in range(max_iter):
        f_bf = f_step_bruteforce(x, T, c, d, V)
        f_cf = f_step_closed_form(x, T, V)
        total += 1
        agree += int(np.array_equal(f_bf, f_cf))
        f = f_bf
        x_new = x_step_lp(f, T, c, d, V)
        if x_new is None:
            break
        if np.allclose(x_new, x, atol=1e-9) and np.array_equal(f_step_bruteforce(x_new, T, c, d, V), f):
            x = x_new
            break
        x = x_new
    f = f_step_bruteforce(x, T, c, d, V)
    return x, f, true_cost(x, f, T, c, d), it + 1, agree, total


# ---------------------------------------------------------------- A. literal energy
def literal_energy_example():
    # one type, two networks: N1 cheap (1.00), N2 expensive (1.50) with 20% discount
    T = np.array([1000.0]); c = np.array([[1.00, 1.50]]); d = np.array([0.0, 0.20])
    V = np.array([np.inf, 500.0])
    rows = []
    for P in [1.0, 3.0, 5.0, 10.0]:
        best = None
        for k in range(2):
            x = np.zeros((1, 2)); x[0, k] = 1
            for fk in [0, 1]:
                f = np.array([0.0, float(fk)])
                if np.any(vol(x, T) + 1e-9 < np.where(np.isinf(V), 0, V) * f):
                    continue
                if f[0] > 0:
                    continue
                C = (T[:, None] * c * x).sum(axis=0)
                E = float(C.sum() - P * (f * d * C).sum())
                cost = true_cost(x, f, T, c, d)
                cand = (E, k, fk, cost)
                if best is None or E < best[0]:
                    best = cand
        rows.append({"P": P, "argmin_network": int(best[1]) + 1, "discount_on": bool(best[2]),
                     "energy": round(best[0], 2), "true_cost_of_argmin": round(best[3], 2)})
    return {"true_optimum": {"network": 1, "cost": 1000.0,
                             "alternative_N2_discounted_cost": 1200.0}, "literal_argmin_by_P": rows}


def main():
    OUT["A_literal_energy"] = literal_energy_example()

    # B + C: random instances, fractional x as in claim 12
    stats = []
    agree_all = 0
    total_all = 0
    for trial in range(200):
        m = int(RNG.integers(5, 40)); n = int(RNG.integers(2, 8))
        T, c, d, V = make_instance(m, n, RNG)
        xg, fg, cg, tg, _ = solve_milp(T, c, d, V, integral_x=False)
        x0a = np.zeros((m, n)); x0a[np.arange(m), c.argmin(axis=1)] = 1  # cheapest base
        x0b = np.full((m, n), 1.0 / n)                                   # uniform split
        best = None
        for x0 in (x0a, x0b):
            x, f, cost, iters, agree, total = claim12(T, c, d, V, x0)
            agree_all += agree; total_all += total
            best = cost if best is None else min(best, cost)
        # multi-start: 10 random feasible initialisations
        ms_best = best
        for _ in range(10):
            x0 = RNG.dirichlet(np.ones(n), size=m)
            x, f, cost, iters, agree, total = claim12(T, c, d, V, x0)
            agree_all += agree; total_all += total
            ms_best = min(ms_best, cost)
        stats.append({"m": m, "n": n, "opt": cg, "claim12_2init": best, "claim12_12starts": ms_best})
    gaps2 = np.array([(s["claim12_2init"] - s["opt"]) / s["opt"] for s in stats])
    gaps12 = np.array([(s["claim12_12starts"] - s["opt"]) / s["opt"] for s in stats])
    OUT["B_fstep_closed_form_agreement"] = {"agree": agree_all, "total": total_all}
    OUT["C_claim12_vs_global"] = {
        "instances": len(stats),
        "two_inits": {"frac_suboptimal(>0.01%)": float(np.mean(gaps2 > 1e-4)),
                      "mean_gap_%": float(100 * gaps2.mean()), "median_gap_%": float(100 * np.median(gaps2)),
                      "p90_gap_%": float(100 * np.quantile(gaps2, 0.9)), "max_gap_%": float(100 * gaps2.max())},
        "twelve_starts": {"frac_suboptimal(>0.01%)": float(np.mean(gaps12 > 1e-4)),
                          "mean_gap_%": float(100 * gaps12.mean()), "median_gap_%": float(100 * np.median(gaps12)),
                          "p90_gap_%": float(100 * np.quantile(gaps12, 0.9)), "max_gap_%": float(100 * gaps12.max())},
    }

    # D: scale test, integral one-network-per-type (patent [0087]-[0091])
    scale = []
    for m, n in [(10, 3), (100, 5), (1000, 10), (5000, 10)]:
        T, c, d, V = make_instance(m, n, RNG)
        x, f, cost, dt, res = solve_milp(T, c, d, V, integral_x=True, time_limit=900)
        scale.append({"types_m": m, "networks_n": n, "binaries": 2 * m * n + n, "status": int(res.status),
                      "mip_gap": float(getattr(res, "mip_gap", float("nan")) or 0.0), "seconds": round(dt, 3)})
    OUT["D_exact_MILP_scale"] = scale

    # E: literal QUBO size (integral x, one-hot + slack-encoded volume inequality)
    qubo = []
    for m, n, vol_total in [(100, 5, 1e6), (1000, 10, 1e7), (10000, 10, 1e8)]:
        b = math.ceil(math.log2(vol_total + 1))
        nvars = m * n + n + n * b
        clique = m + 1 + b
        couplers = m * math.comb(n, 2) + n * math.comb(clique, 2)
        qubo.append({"types_m": m, "networks_n": n, "slack_bits_per_network": b, "logical_vars": nvars,
                     "dense_clique_per_network": clique, "logical_couplers": couplers,
                     "coeff_dynamic_range_T2_over_Tmin": f"~{(vol_total / m) ** 2:.1e} (T_s^2 terms)"})
    OUT["E_literal_QUBO_size"] = {"instances": qubo,
                                  "Advantage2_Z12_max_clique_16m-8": 16 * 12 - 8,
                                  "Advantage2_qubits": "4,400+", "Advantage2_couplers": "40,000+"}

    # concavity of the fractionality penalty beta*x*(1-x)
    OUT["F_fractionality_penalty_hessian"] = "d2/dx2 [beta*x*(1-x)] = -2*beta < 0  (concave for beta>0)"
    print(json.dumps(OUT, indent=1))
    with open(__file__.replace(".py", "_results.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)


if __name__ == "__main__":
    main()
