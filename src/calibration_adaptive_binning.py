"""
Calibration robustness check: equal-mass (adaptive/quantile) binned ECE, compared against the
equal-width-binned ECE already reported in Tables 3-4 and used for H6/H8.

Reviewer comment (Review Comments _Paper-3.docx, "Experimental gaps"): "Calibration assessment
uses 10 equal-width bins (ECE), which can be biased under heavy imbalance; adaptive-binning ECE
... would strengthen claims." This project's own label-distribution table (Table 1) confirms the
premise: threat is 0.33% and severe_toxic is 0.57% of the test set, so equal-width bins can leave
some bins nearly empty for minority labels.

No retraining, no new predictions: recomputes ECE directly from the already-saved
results_proposed/runs/<model>_p4/seed_<seed>/test_outputs.npz (probs, y_true) arrays, using
equal-mass (quantile) bins instead of equal-width bins, otherwise following the same per-label ECE
definition as src/metrics.py's _ece() (Eq. 12 in the manuscript).

Usage: python -m src.calibration_adaptive_binning
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MINORITY = ["threat", "severe_toxic", "identity_hate"]
FREQUENT = ["toxic", "obscene", "insult"]
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


def ece_equal_mass(y, p, bins=10):
    """Expected calibration error, equal-mass (quantile) bins -- same weighted-|gap| definition
    as src/metrics.py's _ece(), but bin edges chosen so each bin holds ~n/bins comments instead
    of covering an equal slice of [0, 1]."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n < bins:
        bins = max(1, n)
    order = np.argsort(p)
    edges = np.array_split(order, bins)
    total = 0.0
    for idx in edges:
        if len(idx) == 0:
            continue
        total += abs(y[idx].mean() - p[idx].mean()) * (len(idx) / n)
    return float(total)


def load_outputs(model, seed):
    d = np.load(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz"))
    return d["probs"], d["y_true"]


def main():
    print("=" * 70)
    print("CALIBRATION ROBUSTNESS CHECK -- equal-mass (adaptive) ECE vs equal-width ECE")
    print("=" * 70)

    rows = []
    for m in MODELS:
        for s in SEEDS:
            probs, y = load_outputs(m, s)
            per_label = {}
            for li, label in enumerate(LABELS):
                per_label[label] = ece_equal_mass(y[:, li], probs[:, li])
            macro = float(np.mean(list(per_label.values())))
            minority = float(np.mean([per_label[l] for l in MINORITY]))
            frequent = float(np.mean([per_label[l] for l in FREQUENT]))
            rows.append({"model": m, "seed": s, "macro_ece_adaptive": macro,
                         "minority_ece_adaptive": minority, "frequent_ece_adaptive": frequent,
                         **{f"{l}_ece_adaptive": per_label[l] for l in LABELS}})
    df = pd.DataFrame(rows)

    print("\n=== Macro ECE (adaptive binning), mean +/- SD over 3 seeds, vs. equal-width (Table 4) ===")
    EQUAL_WIDTH_MACRO_ECE = {  # from Table 4, already in the manuscript, not recomputed here
        "lstm": 0.0400, "bilstm": 0.0389, "gru": 0.0359,
        "bert": 0.0325, "distilbert": 0.0356, "roberta": 0.0360,
    }
    summary_rows = []
    for m in MODELS:
        sub = df[df["model"] == m]
        mean_a, sd_a = sub["macro_ece_adaptive"].mean(), sub["macro_ece_adaptive"].std(ddof=1)
        print(f"  {m:12s} adaptive={mean_a:.4f}+/-{sd_a:.4f}   equal-width={EQUAL_WIDTH_MACRO_ECE[m]:.4f}")
        summary_rows.append({"model": m, "macro_ece_adaptive_mean": mean_a,
                             "macro_ece_adaptive_sd": sd_a,
                             "macro_ece_equal_width_table4": EQUAL_WIDTH_MACRO_ECE[m]})
    summary = pd.DataFrame(summary_rows)

    print("\n=== H8 retest under adaptive binning: minority vs frequent ECE, one-sided paired t-test ===")
    h8_rows = []
    pvals = []
    for m in MODELS:
        sub = df[df["model"] == m]
        minority_vals = sub["minority_ece_adaptive"].values
        frequent_vals = sub["frequent_ece_adaptive"].values
        t, p_two = stats.ttest_rel(minority_vals, frequent_vals)
        p_one = p_two / 2 if t < 0 else 1 - p_two / 2
        pvals.append(p_one)
        h8_rows.append({"model": m, "minority_ece_adaptive_mean": minority_vals.mean(),
                        "frequent_ece_adaptive_mean": frequent_vals.mean(), "p_one_sided": p_one})
    h8 = pd.DataFrame(h8_rows)
    h8["p_holm"] = holm(pvals)
    h8["sig_0.05"] = h8["p_holm"] < 0.05
    print(h8.to_string(index=False))

    df.round(6).to_csv("results_proposed/report/table_calibration_adaptive_binning_raw.csv", index=False)
    summary.round(6).to_csv("results_proposed/report/table_calibration_adaptive_binning_summary.csv", index=False)
    h8.round(6).to_csv("results_proposed/report/table_h8_adaptive_binning_retest.csv", index=False)
    print("\nSaved: table_calibration_adaptive_binning_raw.csv, "
          "table_calibration_adaptive_binning_summary.csv, table_h8_adaptive_binning_retest.csv")
    print(f"\nH8 holds under adaptive binning for {int(h8['sig_0.05'].sum())} / 6 models "
          f"(original equal-width H8 result: 6 / 6 significant).")


if __name__ == "__main__":
    main()
