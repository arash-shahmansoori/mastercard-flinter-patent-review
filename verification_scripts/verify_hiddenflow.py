"""US 2024/0028537 A1 (Hidden Flow Discovery), using the patent's own example ([0037]-[0056]).

Objective [0042]: min |P_x x - P_y y|^2 over binary, column-stochastic P_x (k x n), P_y (k x m),
k = min(m, n). Probability matrix [0053]: F = P_y^T P_x.
"""
import itertools
import json
from fractions import Fraction

import numpy as np

x = np.array([5.431621, 0.647663, 1.76037, 4.640703, 8.930766, 0.469607, 9.180342])
y = np.array([2.673164, 11.410319, 3.185371, 8.0468237, 5.745395])
n, m = len(x), len(y)
k = min(m, n)
out = {"sum_x": round(float(x.sum()), 7), "sum_y": round(float(y.sum()), 7),
       "sum_residual": float(x.sum() - y.sum())}


def energy(route_x, route_y):
    vx = np.zeros(k); vy = np.zeros(k)
    for i, r in enumerate(route_x):
        vx[r] += x[i]
    for j, r in enumerate(route_y):
        vy[r] += y[j]
    return float(((vx - vy) ** 2).sum())


def F_matrix(route_x, route_y):
    Px = np.zeros((k, n)); Py = np.zeros((k, m))
    for i, r in enumerate(route_x):
        Px[r, i] = 1
    for j, r in enumerate(route_y):
        Py[r, j] = 1
    return Py.T @ Px


# patent's reported solution: A={x1,x2,x4}->{y1,y4}, B={x3,x6,x7}->{y2}, C={x5}->{y3,y5}
pat_x = [0, 0, 1, 0, 2, 1, 1]
pat_y = [0, 1, 2, 0, 2]
triv_x = [0] * n
triv_y = [0] * m
out["energy_patent_solution"] = energy(pat_x, pat_y)
out["energy_trivial_single_route"] = energy(triv_x, triv_y)
F = F_matrix(pat_x, pat_y)
out["F_from_formula_[0053]_is_binary"] = bool(set(np.unique(F)) <= {0.0, 1.0})
out["F_formula_row_y1"] = F[0].tolist()
out["F_reported_[0055]_row_y1"] = ["1/3", "1/3", "0", "1/3", "0", "0", "0"]

# exhaustive search over all labelled route assignments (k^(n+m) = 5^12 ~ 2.4e8 is too many);
# enumerate set partitions instead: blocks must balance, then count labelled embeddings.
tol = 1e-6
blocks = [((0, 1, 3), (0, 3)), ((2, 5, 6), (1,)), ((4,), (2, 4))]
# verify these are the only minimal balanced blocks by brute force over subsets
minimal = []
for rx in range(1, n + 1):
    for Sx in itertools.combinations(range(n), rx):
        sx = x[list(Sx)].sum()
        for ry in range(1, m + 1):
            for Sy in itertools.combinations(range(m), ry):
                if abs(sx - y[list(Sy)].sum()) < tol:
                    minimal.append((Sx, Sy))
# keep only those not decomposable into two balanced sub-blocks
def decomposable(Sx, Sy):
    for rx in range(1, len(Sx)):
        for A in itertools.combinations(Sx, rx):
            sa = x[list(A)].sum()
            for ry in range(1, len(Sy)):
                for B in itertools.combinations(Sy, ry):
                    if abs(sa - y[list(B)].sum()) < tol:
                        return True
    return False
atoms = [b for b in minimal if not decomposable(*b)]
out["balanced_atomic_blocks_found_by_bruteforce"] = [[list(a), list(b)] for a, b in atoms]

# interpretations = set partitions of the atoms (Bell(3) = 5); labelled ground states:
def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p
interps = list(set_partitions(list(range(len(atoms)))))
labelled = 0
link = np.zeros((m, n), dtype=object)
for i in range(m):
    for j in range(n):
        link[i, j] = Fraction(0)
for p in interps:
    b = len(p)
    perms = 1
    for t in range(b):
        perms *= (k - t)
    labelled += perms
    for grp in p:
        xs = [ii for a in grp for ii in atoms[a][0]]
        ys = [jj for a in grp for jj in atoms[a][1]]
        for jj in ys:
            for ii in xs:
                link[jj, ii] += Fraction(1, len(interps))
out["number_of_interpretations"] = len(interps)
out["number_of_degenerate_zero_energy_labelled_ground_states"] = labelled
out["boltzmann_link_probability_row_y1"] = [str(v) for v in link[0]]
out["boltzmann_link_probability_row_y2"] = [str(v) for v in link[1]]
qubo_vars = k * (n + m)
out["qubo_variables_example"] = qubo_vars
for nin, nout in [(50, 50), (300, 300)]:
    kk = min(nin, nout)
    out[f"qubo_variables_{nin}in_{nout}out"] = kk * (nin + nout)
print(json.dumps(out, indent=1))
with open(__file__.replace(".py", "_results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
