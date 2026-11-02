"""
H2, DeLong's test re-combined via Fisher's method instead of median-across-labels.

Reviewer comment (Review Comments _Paper-3.docx, "Experimental evaluation assessment"):
"for DeLong, using a meta-analytic combination (e.g., Fisher/Stouffer) across labels rather than
median-p would preserve more information". src/h2_full_methods.py's delong_per_pair() combines
the 3 per-seed p-values for each label by taking their median, then the manuscript's Appendix C
(Table A6) reports that per-label median again summarised by... the median across the 6 labels.
This script re-derives the same 9 cross-family pairs using Fisher's combined-probability test
(the same method already used for H5 in this paper, scipy.stats.combine_pvalues(method="fisher")),
combining all 6 labels x 3 seeds = 18 independent per-run DeLong p-values per pair in one Fisher
test, Holm-corrected over the 9 pairs. No retraining: reads the same saved
results_proposed/runs/<model>_p4/seed_<seed>/test_outputs.npz files as h2_full_methods.py.

Usage: python -m src.h2_delong_fisher
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.h2_full_methods import RECURRENT, TRANSFORMER, SEEDS, LABELS, PAIRS, holm, load_outputs, delong_test


def delong_fisher_per_pair():
    rows = []
    for rec, trans in PAIRS:
        all_p = []
        per_label_seed = []
        for label_idx, label in enumerate(LABELS):
            for s in SEEDS:
                probs_r, y_r = load_outputs(rec, s)
                probs_t, y_t = load_outputs(trans, s)
                assert np.array_equal(y_r, y_t)
                _, _, p = delong_test(probs_t[:, label_idx], probs_r[:, label_idx], y_r[:, label_idx])
                p = float(np.clip(p, 1e-300, 1.0))  # Fisher's method needs p > 0
                all_p.append(p)
                per_label_seed.append({"label": label, "seed": s, "p": p})
        fisher_stat, fisher_p = stats.combine_pvalues(all_p, method="fisher")
        rows.append({
            "recurrent": rec, "transformer": trans,
            "n_pvalues_combined": len(all_p),
            "fisher_statistic": fisher_stat, "fisher_p": fisher_p,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("H2, DeLong's test, Fisher-combined across 6 labels x 3 seeds (18 p-values/pair)")
    print("Reviewer-suggested alternative to the median-across-labels combination")
    print("=" * 70)
    df = delong_fisher_per_pair()
    df["p_holm_9pairs"] = holm(df["fisher_p"].tolist())
    df["sig_0.05"] = df["p_holm_9pairs"] < 0.05
    print(df.to_string(index=False))
    df.round(6).to_csv("results_proposed/report/table_h2_delong_fisher.csv", index=False)
    print("\nSaved: results_proposed/report/table_h2_delong_fisher.csv")
    print(f"\nSignificant pairs (Holm, alpha=0.05): {int(df['sig_0.05'].sum())} / 9")
    print("For comparison, the original median-across-labels DeLong result (Table A6): 0 / 9")


if __name__ == "__main__":
    main()
