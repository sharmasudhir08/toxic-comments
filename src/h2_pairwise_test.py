"""
H2: each transformer exceeds each recurrent model in macro ROC-AUC / macro-F1(M).

Pre-registered spec (pre_registration.md Sec. 4.1): "One-way ANOVA + Tukey HSD; DeLong per label
(median p over seeds); paired bootstrap over test comments; Holm over the 9 pairs."

What this script actually computes: a paired t-test across the 3 matched seeds for each of the 9
cross-family tuned-threshold pairs, Holm-corrected over exactly those 9 comparisons. This is a
DEVIATION from the pre-registered Tukey HSD + DeLong method -- flagged explicitly, not silently
substituted. Tukey HSD and DeLong's test were not implemented as of 1 Nov 2026; this script exists
so the 9-pair Holm correction used in paper-3_manuscript.docx is a saved, traceable output rather
than an ephemeral calculation, pending a decision on whether to implement the exact pre-registered
test.

No new training. Reads results_proposed/runs/<model>_p4/seed_<seed>/metrics.json.

Usage: python -m src.h2_pairwise_test
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"


def holm(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(idx):
        c = pvals[i] * (n - rank)
        running = max(running, c)
        adj[i] = min(running, 1.0)
    return adj


def load(model, seed):
    path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test"]


def main():
    print("=" * 70)
    print("H2: pairwise recurrent-vs-transformer comparison (9 cross-family pairs)")
    print("DEVIATION FROM PRE-REGISTRATION: paired t-test used, not Tukey HSD + DeLong.")
    print("=" * 70)

    pairs = [(r, t) for r in RECURRENT for t in TRANSFORMER]
    for metric, label in [("f1_macro_m", "macro-F1(M)"), ("roc_auc_macro", "macro ROC-AUC")]:
        rows = []
        for rec, trans in pairs:
            rec_vals = np.array([load(rec, s)[metric] for s in SEEDS])
            trans_vals = np.array([load(trans, s)[metric] for s in SEEDS])
            t, p = stats.ttest_rel(trans_vals, rec_vals)
            rows.append({"recurrent": rec, "transformer": trans, "metric": metric,
                         "diff_trans_minus_rec": trans_vals.mean() - rec_vals.mean(), "p_raw": p})
        pvals = [r["p_raw"] for r in rows]
        padj = holm(pvals)
        for r, pa in zip(rows, padj):
            r["p_holm_9pairs"] = pa
            r["sig_0.05"] = pa < 0.05

        df = pd.DataFrame(rows)
        out = f"results_proposed/report/table_h2_pairwise_{metric}.csv"
        df.round(5).to_csv(out, index=False)
        print(f"\n{label}:")
        print(df[["recurrent", "transformer", "diff_trans_minus_rec", "p_raw", "p_holm_9pairs", "sig_0.05"]]
              .to_string(index=False))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
