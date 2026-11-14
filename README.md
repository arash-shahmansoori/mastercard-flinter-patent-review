# Technical Critique of the Mastercard Patent Portfolio Naming Stephen Patrick Flinter

Full 19-page technical review (`Mastercard_Patent_Portfolio_Critical_Review.pdf` and LaTeX source) and standalone verification scripts auditing **15 patent families** (33 published patent documents, 2015–2026) naming **Stephen Patrick Flinter** as an inventor with 32 co-inventors, assigned to **Mastercard International Incorporated**.

- **Reviewer:** Arash Shahmansoori
- **Report:** [`Mastercard_Patent_Portfolio_Critical_Review.pdf`](Mastercard_Patent_Portfolio_Critical_Review.pdf) (19 pages)
- **Granted US patents reviewed:** `US 12,363,024 B2` (*Switching optimization*, granted 15 Jul 2025) and `US 12,718,235 B2` (*Transaction analysis and visualisation*, granted 25 Aug 2026)
- **Pending families reviewed:** 11 live families across the USPTO, EPO, WIPO (PCT), CN, JP, KR, and AU (plus 2 abandoned families documented in Table 1)

## Environment & Setup

Tested with Python 3.12, `numpy`, `scipy >= 1.11` (HiGHS MILP/LP solver; `1.16` used in the report), and `scikit-learn` on a laptop CPU (`Intel i7-7700HQ`). No proprietary data is required.

```bash
pip install -r requirements.txt
```

## Reproducing All Numerical Results

Run the scripts from inside `verification_scripts/`:

```bash
cd verification_scripts
python3 verify_switching.py
python3 verify_switching_fractional_scale.py
python3 verify_allocation.py
python3 verify_hiddenflow.py
python3 verify_ml.py
```

## What Each Script Reproduces

| Script | Report sections / filings | Output file | What it verifies |
|---|---|---|---|
| `verify_switching.py` | Issue 1 (`US 12,363,024 B2`) | `verify_switching_results.json` | Literal `P`-scaled discount energy picking a 20% costlier network for `P > 5/3`; closed-form threshold rule matching `2^n` enumeration in **5,496 / 5,496** iterations; Claim-12 alternation vs. global MILP across 200 instances (**19.5%–23.0%** suboptimal, up to **24.4%** gap); exact HiGHS MILP scaling to **100,010** binaries; literal QUBO clique/coupler counts vs. D-Wave Advantage2 (`Z_12`); concavity of `beta * x * (1 - x)`. |
| `verify_switching_fractional_scale.py` | Issue 1.5 (`US 12,363,024 B2`) | `verify_switching_fractional_scale_results.json` | Exact fractional-routing solution via `2^10 = 1,024` discount-pattern enumeration + 1 LP per pattern (`12 s` at `m = 200`, `54 s` at `m = 1,000`). |
| `verify_allocation.py` | Issue 6 (`US 2024/0037582 A1`) | `verify_allocation_results.json` | Total unimodularity / LP integrality of the disclosed count-budget card-offer allocation model with disjoint exclusivity groups: **0 fractional entries** at `80,000` (`1.05 s`), `1,000,000` (`21.25 s`), and `2,400,000` binary variables (`70.98 s`). |
| `verify_hiddenflow.py` | Issue 7 (`US 2024/0028537 A1`) | `verify_hiddenflow_results.json` | Recomputes the patent's 7-input, 5-output worked example (`[0037]`–`[0056]`): identical `4.9e-13` energy of the uninformative single-route assignment, `Bell(3) = 5` interpretations (**125** labelled zero-energy ground states), binary `F = P_y^T P_x` vs. `1/3` entries in `[0055]`, and exact Boltzmann link probabilities. |
| `verify_ml.py` | Issues 3, 4 & 5 (`US 2024/0420147 A1`, `US 2026/0065141 A1`, `US 2026/0037961 A1`) | `verify_ml_results.json` | Algebraic inversion of batch-detection prevalence (`pi = 0.4995`, `[0.4969, 0.5022]`); Q1 2022 fraud trivial baselines (`0.908` all-licit accuracy) and false-flag rate (`43.1 -> 89.5` per `10k` txs); test-set candidate selection optimism (`K` up to `1,100`); Claim 4 random-forest MDI ranking high-cardinality noise (`address_hash`, `block_height`) above the real equal-output CoinJoin signal. |

The committed `*_results.json` files contain the reference outputs reported in the PDF.

## Disclaimer

This repository presents independent technical and scientific criticism of publicly published patent documents and academic literature. References to statutory patentability standards (`35 U.S.C. §§ 101, 102, 103, 112`; `Arts. 52, 54, 56, 83, 84 EPC`) explain how the technical findings relate to those criteria and do not constitute legal advice.
