"""Exact solve of the claim-12 problem class by enumerating all 2^n discount patterns.

With the binary discount vector f fixed, the continuous routing problem is an LP, so
enumerating f in {0,1}^n and keeping the best feasible LP is an exact algorithm.
"""
import itertools
import json
import time

import numpy as np

from verify_switching import make_instance, true_cost, x_step_lp

rng = np.random.default_rng(7)
rows = []
for m, n in [(200, 10), (1000, 10)]:
    T, c, d, V = make_instance(m, n, rng)
    t0 = time.perf_counter()
    best, nlp = None, 0
    for bits in itertools.product([0, 1], repeat=n):
        f = np.array(bits, float)
        if (V * f).sum() > T.sum() * n:  # cheap pre-filter, never binding here
            continue
        x = x_step_lp(f, T, c, d, V)
        nlp += 1
        if x is None:
            continue
        cost = true_cost(x, f, T, c, d)
        if best is None or cost < best:
            best = cost
    dt = time.perf_counter() - t0
    rows.append({"types_m": m, "networks_n": n, "patterns_enumerated": 2 ** n, "lps_solved": nlp,
                 "seconds_total": round(dt, 1), "seconds_per_lp": round(dt / nlp, 4)})
    print(rows[-1], flush=True)
with open(__file__.replace(".py", "_results.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
