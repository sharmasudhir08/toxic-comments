"""
Verification script (3 Nov 2026): checks the paper's claim "No pairwise difference within
either family alone reaches significance on either metric" (Sec. 4.4) and "Within the recurrent
family, no model is significantly ahead of another on either metric" (Sec. 5.1 Discussion).

Reuses the exact same Tukey HSD computation as src/h2_full_methods.py (one-way ANOVA across
all six models, three seeds each), but reports the 6 WITHIN-family pairs (3 recurrent-internal +
3 transformer-internal) instead of the 9 cross-family pairs that script saves. No new training;
reads already-saved results_proposed/runs/<model>_p4/seed_<seed>/metrics.json only.

Usage: python -m src.h2_within_family_check
"""
import json
import os
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
MODELS = RECURRENT + TRANSFORMER
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"

WITHIN_PAIRS = [
    ("lstm", "bilstm"), ("lstm", "gru"), ("bilstm", "gru"),
    ("bert", "distilbert"), ("bert", "roberta"), ("distilbert", "roberta"),
]


def load_metrics(model, seed):
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["test"]


def tukey_hsd_within(metric):
    labels, values = [], []
    for m in MODELS:
        for s in SEEDS:
            labels.append(m)
            values.append(load_metrics(m, s)[metric])
    res = pairwise_tukeyhsd(np.array(values), np.array(labels), alpha=0.05)
    df = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    sub = df[df.apply(lambda r: (r["group1"], r["group2"]) in WITHIN_PAIRS or (r["group2"], r["group1"]) in WITHIN_PAIRS, axis=1)]
    return sub.copy()


if __name__ == "__main__":
    print("=" * 70)
    print("WITHIN-FAMILY Tukey HSD check (3 recurrent-internal + 3 transformer-internal pairs)")
    print("=" * 70)
    for metric in ["roc_auc_macro", "f1_macro_m"]:
        print(f"\n--- {metric} ---")
        sub = tukey_hsd_within(metric)
        print(sub.to_string(index=False))
        out_path = f"results_proposed/report/table_h2_within_family_tukey_{metric}.csv"
        sub.to_csv(out_path, index=False)
        print(f"saved: {out_path}")
        n_sig = sub["reject"].sum()
        print(f"within-family pairs reaching significance (reject=True): {n_sig} of {len(sub)}")
