"""Arithmetic and simulation checks for the blockchain-ML applications.

(a) US 2026/0037961 A1 [0083]: infer evaluation prevalence from accuracy/precision/recall.
(b) US 2024/0420147 A1 App. IV-V: trivial baselines, consistency, operational burden.
(c) US 2026/0065141 A1 claims 3, 5, 6 / [0032]-[0035], [0057]: optimism of selecting the
    best of K candidate models on the same test set that is then reported.
(d) US 2026/0065141 A1 claim 4: impurity-based (MDI) importance ranks pure-noise,
    high-cardinality features (e.g. block height) above weak real signal.
"""
import json

import numpy as np
from scipy.stats import norm

OUT = {}


def prevalence_from_metrics(acc, p, r):
    # 1 - acc = pi * [(1 - r) + r (1 - p) / p]
    return (1 - acc) / ((1 - r) + r * (1 - p) / p)


# (a) batched transactions: acc 0.9813, precision 0.9931, recall 0.9693 (4-decimal rounding)
grid = [prevalence_from_metrics(a, p, r)
        for a in (0.98125, 0.98135) for p in (0.99305, 0.99315) for r in (0.96925, 0.96935)]
OUT["a_batched_inferred_prevalence"] = {
    "point": round(prevalence_from_metrics(0.9813, 0.9931, 0.9693), 4),
    "range_under_rounding": [round(min(grid), 4), round(max(grid), 4)]}

# (b) fraud prediction appendices
months = {"Jan-2022": (8_233_012, 835_855), "Feb-2022": (6_505_435, 622_706), "Mar-2022": (7_964_328, 731_425)}
OUT["b_prevalence_and_trivial_accuracy"] = {
    k: {"illicit_share_%": round(100 * b / a, 2), "all_licit_accuracy": round(1 - b / a, 4)} for k, (a, b) in months.items()}
pi_test = 731_425 / 7_964_328
models = {"initial": (0.81, 0.20), "updated": (0.80, 0.39)}
res = {}
for name, (p, r) in models.items():
    err = pi_test * ((1 - r) + r * (1 - p) / p)
    fp_rate = pi_test * r * (1 - p) / p
    res[name] = {"implied_accuracy_on_Mar": round(1 - err, 4),
                 "false_flags_per_10k_tx": round(1e4 * fp_rate, 1),
                 "missed_illicit_per_10k_tx": round(1e4 * pi_test * (1 - r), 1),
                 "f1_recomputed": round(2 * p * r / (p + r), 3)}
OUT["b_fraud_model_consistency_and_burden"] = res
OUT["b_ratio_to_chainalysis_2022_value_share_0.24%"] = round((731_425 / 7_964_328) / 0.0024, 1)

# (c) test-set selection optimism (Gaussian approximation + simulation)
def expected_max_std_normal(K, sims=20000, rng=np.random.default_rng(0)):
    return float(rng.standard_normal((sims, K)).max(axis=1).mean())

sel = []
for prev, ntest in [(0.001, 1_000_000), (0.02, 1_000_000)]:
    npos = int(prev * ntest)
    fnr_true = 0.02
    sd = np.sqrt(fnr_true * (1 - fnr_true) / npos)
    for K in [10, 100, 1100]:
        em = expected_max_std_normal(K)
        for rho in [0.0, 0.9]:
            optimism = sd * np.sqrt(1 - rho) * em
            sel.append({"test_prevalence": prev, "positives_in_test": npos, "candidates_K": K, "rho": rho,
                        "true_FNR_%": 100 * fnr_true, "expected_reported_best_FNR_%": round(100 * (fnr_true - optimism), 3),
                        "optimism_pp": round(100 * optimism, 3)})
OUT["c_test_set_selection_optimism"] = sel

# (d) MDI vs permutation importance on synthetic MAT-like data
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
N = 60_000
n_equal_out = rng.poisson(0.6, N)                       # structural signal
io_ratio = rng.lognormal(0, 0.6, N)                     # weak signal
block_height = rng.integers(700_000, 860_000, N)       # pure noise, high cardinality
addr_hash_num = rng.random(N)                           # pure noise, continuous
is_segwit = rng.integers(0, 2, N)                       # pure noise, binary
logit = -4.2 + 1.6 * n_equal_out + 0.35 * np.log(io_ratio)
y = rng.random(N) < 1 / (1 + np.exp(-logit))
X = np.column_stack([n_equal_out, io_ratio, block_height, addr_hash_num, is_segwit]).astype(float)
names = ["n_equal_outputs (signal)", "input_output_ratio (weak signal)", "block_height (noise)",
         "address_hash_numeric (noise)", "binary_flag (noise)"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
mdi = rf.feature_importances_
perm = permutation_importance(rf, Xte, yte, scoring="average_precision", n_repeats=10, random_state=0, n_jobs=-1)
OUT["d_mdi_vs_permutation"] = {
    "prevalence": round(float(y.mean()), 4),
    "features": [{"feature": nm, "MDI_share_%": round(100 * float(a), 1),
                  "heldout_permutation_dAP": round(float(b), 4)} for nm, a, b in zip(names, mdi, perm.importances_mean)],
    "mdi_rank": [names[i] for i in np.argsort(-mdi)]}
print(json.dumps(OUT, indent=1))
with open(__file__.replace(".py", "_results.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
