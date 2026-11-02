"""
Calibration analysis (ECE, Brier) across all six models, both families.

Reads already-saved results_proposed/runs/<model>_p4/seed_<seed>/metrics.json files
(no new training, no new predictions -- ece_macro/brier_macro/per_label ece/brier are
already computed and saved by src/metrics.py for every run).

Computes, per model (mean +/- SD over available seeds):
  - macro ECE, macro Brier
  - per-label ECE, split into minority (threat/severe_toxic/identity_hate) vs
    frequent (toxic/obscene/insult) groups
  - a one-sided paired t-test (minority ECE < frequent ECE, i.e. minority better
    calibrated), Holm-corrected across models with >=2 seeds

Models with fewer than the expected 3 seeds are reported with whatever seeds exist,
clearly flagged as partial -- never silently padded or estimated.

Usage: python -m src.calibration_analysis
"""
import json
import os
import numpy as np
from scipy import stats

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MINORITY = ["threat", "severe_toxic", "identity_hate"]
FREQUENT = ["toxic", "obscene", "insult"]

RESULTS_DIR = "results_proposed/runs"


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


def load_model_data(model):
    """Returns dict: seed -> test metrics dict, only for seeds that exist."""
    out = {}
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                out[seed] = json.load(f)["test"]
    return out


def main():
    print("=" * 70)
    print("CALIBRATION ANALYSIS -- all six models, both families")
    print("=" * 70)

    data = {m: load_model_data(m) for m in MODELS}

    print("\nSeed completeness:")
    for m in MODELS:
        seeds_present = sorted(data[m].keys())
        flag = "" if len(seeds_present) == 3 else "  <-- PARTIAL, not 3 seeds"
        print(f"  {m:12s} seeds={seeds_present}{flag}")

    print("\n=== Macro ECE / Brier, mean +/- SD over available seeds ===")
    model_ece = {}
    for m in MODELS:
        seeds_present = sorted(data[m].keys())
        if not seeds_present:
            print(f"  {m:12s} NO DATA")
            continue
        ece = np.array([data[m][s]["ece_macro"] for s in seeds_present])
        brier = np.array([data[m][s]["brier_macro"] for s in seeds_present])
        model_ece[m] = ece
        sd_ece = ece.std(ddof=1) if len(ece) > 1 else float("nan")
        sd_brier = brier.std(ddof=1) if len(brier) > 1 else float("nan")
        n_note = "" if len(seeds_present) == 3 else f" (n={len(seeds_present)}, partial)"
        print(f"  {m:12s} ECE={ece.mean():.4f}+/-{sd_ece:.4f}  Brier={brier.mean():.4f}+/-{sd_brier:.4f}{n_note}")

    print("\n=== Per-label ECE: minority vs frequent, per model (mean over available seeds) ===")
    minority_freq_vals = {}
    for m in MODELS:
        seeds_present = sorted(data[m].keys())
        if not seeds_present:
            continue
        min_per_seed = []
        freq_per_seed = []
        for s in seeds_present:
            pl = data[m][s]["per_label"]
            min_per_seed.append(np.mean([pl[l]["ece"] for l in MINORITY]))
            freq_per_seed.append(np.mean([pl[l]["ece"] for l in FREQUENT]))
        minority_freq_vals[m] = (np.array(min_per_seed), np.array(freq_per_seed))
        min_ece = np.mean(min_per_seed)
        freq_ece = np.mean(freq_per_seed)
        ratio = min_ece / freq_ece if freq_ece else float("nan")
        n_note = "" if len(seeds_present) == 3 else f" (n={len(seeds_present)}, partial)"
        print(f"  {m:12s} minority_ECE={min_ece:.4f}  frequent_ECE={freq_ece:.4f}  ratio={ratio:.2f}x{n_note}")

    print("\n=== Minority vs frequent ECE: one-sided paired t-test (minority < frequent), per model ===")
    print("    (models with <2 seeds cannot be tested -- reported as n/a, not estimated)")
    testable = [m for m in MODELS if len(data[m]) >= 2]
    pvals, diffs, names = [], [], []
    for m in testable:
        min_vals, freq_vals = minority_freq_vals[m]
        t, p = stats.ttest_rel(min_vals, freq_vals, alternative="less")
        pvals.append(p)
        diffs.append(min_vals.mean() - freq_vals.mean())
        names.append(m)
    adj = holm_correct(pvals) if pvals else []
    for m, p, padj, diff in zip(names, pvals, adj, diffs):
        n_note = "" if len(data[m]) == 3 else f" (n={len(data[m])}, partial -- power is limited)"
        print(f"  {m:12s} diff(minority-frequent)={diff:+.4f}  p={p:.4f}  p_Holm={padj:.4f}{n_note}")
    for m in MODELS:
        if m not in testable and m in data and data[m]:
            print(f"  {m:12s} n/a (only {len(data[m])} seed(s), cannot run a paired test)")

    print("\n=== Does the best-F1 model match the best-calibrated model? ===")
    for m in MODELS:
        if m not in data or not data[m]:
            continue
        seeds_present = sorted(data[m].keys())
        f1 = np.mean([data[m][s]["f1_macro_m"] for s in seeds_present])
        ece = model_ece[m].mean()
        n_note = "" if len(seeds_present) == 3 else f" (n={len(seeds_present)}, partial)"
        print(f"  {m:12s} F1_macro_M={f1:.4f}  ECE={ece:.4f}{n_note}")

    missing = [m for m in MODELS if len(data[m]) < 3]
    if missing:
        print(f"\nNOTE: {missing} do not have all 3 seeds yet. Family-level (6-model) conclusions")
        print("should not be drawn from this run until every model has 3 complete seeds --")
        print("re-run this script once they do.")


if __name__ == "__main__":
    main()
