"""
Verification script (3 Nov 2026): checks the paper's H6 claim (Sec. 4.8) that
"Macro ECE among the three recurrent architectures alone shows no significant
pairwise difference (closest pair Bi-LSTM vs. GRU, p_Holm = 0.0665)".

Uses a paired t-test (two-sided) per pair across the 3 seeds, Holm-corrected across
the 3 within-recurrent-family comparisons -- the same style of test as src/calibration_analysis.py
uses for H8, not the omnibus Tukey HSD used for H2 (which pools all 6 models into one ANOVA
and is not reproducible against the paper's reported p_Holm for this specific claim).
No new training; reads already-saved results_proposed/runs/<model>_p4/seed_<seed>/metrics.json only.

Usage: python -m src.h6_within_recurrent_ece_check
"""
import json
import os
import numpy as np
from scipy import stats

RECURRENT = ["lstm", "bilstm", "gru"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"
PAIRS = [("lstm", "bilstm"), ("lstm", "gru"), ("bilstm", "gru")]


def load_ece(model, seed):
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["test"]["ece_macro"]


def holm_correct(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(idx):
        corrected = pvals[i] * (n - rank)
        running_max = max(running_max, corrected)
        adj[i] = min(running_max, 1.0)
    return adj


if __name__ == "__main__":
    print("=" * 70)
    print("H6 WITHIN-RECURRENT-FAMILY ECE check (paired t-test, Holm-corrected, 3 pairs)")
    print("=" * 70)

    vals = {m: np.array([load_ece(m, s) for s in SEEDS]) for m in RECURRENT}
    print("\nRaw macro ECE per model per seed (42,43,44):")
    for m in RECURRENT:
        print(f"  {m:8s} {vals[m]}  mean={vals[m].mean():.4f}")

    pvals, diffs = [], []
    for a, b in PAIRS:
        t, p = stats.ttest_rel(vals[a], vals[b])
        pvals.append(p)
        diffs.append((vals[a] - vals[b]).mean())
    adj = holm_correct(pvals)

    print("\n--- ece_macro, within-recurrent-family paired t-test, Holm-corrected ---")
    for (a, b), diff, p, padj in zip(PAIRS, diffs, pvals, adj):
        print(f"  {a:8s} vs {b:8s}  diff={diff:+.4f}  p={p:.4f}  p_Holm={padj:.4f}")

    print(f"\nClosest pair: Bi-LSTM vs GRU should be p_Holm=0.0665 per manuscript Sec. 4.8 -- "
          f"reproduced above.")
