"""US 2024/0037582 A1 (Allocation optimization): the patent's own model is an LP.

Model ([0070]-[0076]): maximise sum r_ij x_ij, x binary, per-cardholder cap
sum_i x_ij <= k, per-offer count budget b_i^L <= sum_j x_ij <= b_i^U, plus (claim 4)
mutually exclusive offer groups (at most one offer per group per cardholder).
Offer totals form a partition; cardholder totals and (group, cardholder) totals
form a laminar family, so the constraint matrix is the incidence matrix of two
laminar families and is totally unimodular. The LP relaxation is therefore integral.
"""
import json
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

rng = np.random.default_rng(11)
rows_out = []
for n_offers, n_card, k, n_groups in [(40, 2_000, 3, 8), (100, 10_000, 5, 20), (120, 20_000, 8, 30)]:
    m, n = n_offers, n_card
    N = m * n
    r = rng.beta(0.6, 6.0, size=(m, n))
    group = rng.integers(0, n_groups, size=m)
    cap_total = n * min(k, n_groups)
    bU = rng.integers(max(2, cap_total // (4 * m)), max(3, cap_total // m), size=m).astype(float)
    bL = np.floor(0.5 * bU)
    var = np.arange(N).reshape(m, n)
    # offer rows: one row per offer
    A_off = csr_matrix((np.ones(N), (np.repeat(np.arange(m), n), var.ravel())), shape=(m, N))
    # cardholder rows
    A_card = csr_matrix((np.ones(N), (np.tile(np.arange(n), m), var.ravel())), shape=(n, N))
    # exclusivity rows: (group g, cardholder j) for groups with >= 2 offers
    gids = [g for g in range(n_groups) if np.sum(group == g) >= 2]
    gmap = {g: t for t, g in enumerate(gids)}
    sel = np.isin(group, gids)
    offers_sel = np.where(sel)[0]
    rr = (np.array([gmap[g] for g in group[offers_sel]])[:, None] * n + np.arange(n)[None, :]).ravel()
    cc = var[offers_sel].ravel()
    A_exc = csr_matrix((np.ones(len(cc)), (rr, cc)), shape=(len(gids) * n, N))
    A_ub = vstack([A_off, -A_off, A_card, A_exc]).tocsr()
    b_ub = np.concatenate([bU, -bL, np.full(n, float(k)), np.ones(len(gids) * n)])
    t0 = time.perf_counter()
    res = linprog(-r.ravel(), A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs")
    dt = time.perf_counter() - t0
    xs = res.x
    frac = int(np.sum((xs > 1e-7) & (xs < 1 - 1e-7)))
    rows_out.append({"offers": m, "cardholders": n, "cap_k": k, "exclusive_groups": len(gids),
                     "binary_vars": N, "constraints": int(A_ub.shape[0]), "lp_status": int(res.status),
                     "fractional_entries_in_LP_optimum": frac, "seconds": round(dt, 2)})
    print(rows_out[-1], flush=True)

patent_scale = {"offers": 1_000, "cardholders": 1_000_000, "binary_vars": 10 ** 9,
                "slack_bits_if_cap_k_le_15": 4 * 1_000_000, "largest_annealer_qubits": "~4,400-5,000"}
out = {"lp_integrality": rows_out, "patent_stated_scale": patent_scale}
with open(__file__.replace(".py", "_results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
