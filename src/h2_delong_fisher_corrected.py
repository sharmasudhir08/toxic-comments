"""
Rebuilds Table A7 (DeLong's test, macro ROC-AUC, re-combined with Fisher's method across
all 18 per-label-per-seed p-values per pair) using the CORRECTED (grouped-midrank) DeLong
implementation from src/h2_full_methods.py, which src/h2_delong_fisher.py's own saved
table (results_proposed/report/table_h2_delong_fisher.csv) does not use -- that file
still reflects the pre-fix, ungrouped-midrank DeLong p-values (near-uniform p=1.0 after
Holm, matching the submitted paper's current Table A7).

No new training or predictions: reads only already-saved test_outputs.npz files, via
src.h2_full_methods.delong_per_pair(), which already computes per-seed DeLong p-values
per (pair, label) -- this script only adds the Fisher combination and Holm correction
across pairs, which that function does not itself do.

Usage: python -m src.h2_delong_fisher_corrected
"""
import csv
import numpy as np
from scipy import stats

from src.h2_full_methods import delong_per_pair, holm, PAIRS


def fisher_combine(pvals):
    pvals = np.clip(pvals, 1e-300, 1.0)
    stat = -2 * np.sum(np.log(pvals))
    p = stats.chi2.sf(stat, df=2 * len(pvals))
    return stat, p


def main():
    df = delong_per_pair()
    rows = []
    for rec, trans in PAIRS:
        sub = df[(df["recurrent"] == rec) & (df["transformer"] == trans)]
        all_ps = [p for ps in sub["seed_ps"] for p in ps]
        assert len(all_ps) == 18, len(all_ps)
        stat, p = fisher_combine(all_ps)
        rows.append({"recurrent": rec, "transformer": trans, "n_pvalues_combined": 18,
                     "fisher_statistic": stat, "fisher_p": p})
    pvals = [r["fisher_p"] for r in rows]
    adj = holm(pvals)
    for r, a in zip(rows, adj):
        r["p_holm_9pairs"] = a
        r["sig_0.05"] = a < 0.05

    out = "results_v2/table_a7_delong_fisher_corrected.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    for r in rows:
        print(r["recurrent"], r["transformer"], round(r["fisher_statistic"], 3),
              round(r["fisher_p"], 6), round(r["p_holm_9pairs"], 6), r["sig_0.05"])


if __name__ == "__main__":
    main()
