"""
Saves the aggregate per-model results of src.calibration_analysis's one-sided
paired t-test (minority ECE < frequent ECE) to results_v2/, for traceability.
No new computation beyond what src.calibration_analysis already prints to
console; reads the same already-saved results_proposed/runs/*_p4/seed_*/metrics.json
files, no new training or predictions.

Usage: python -m src.h8_raw_ece_paired_ttest_save
"""
import csv
import numpy as np
from scipy import stats

from src.calibration_analysis import MODELS, load_model_data, MINORITY, FREQUENT, holm_correct


def main():
    data = {m: load_model_data(m) for m in MODELS}
    rows = []
    pvals, diffs, names, min_eces, freq_eces = [], [], [], [], []
    for m in MODELS:
        seeds_present = sorted(data[m].keys())
        min_per_seed, freq_per_seed = [], []
        for s in seeds_present:
            pl = data[m][s]["per_label"]
            min_per_seed.append(np.mean([pl[l]["ece"] for l in MINORITY]))
            freq_per_seed.append(np.mean([pl[l]["ece"] for l in FREQUENT]))
        min_vals, freq_vals = np.array(min_per_seed), np.array(freq_per_seed)
        t, p = stats.ttest_rel(min_vals, freq_vals, alternative="less")
        pvals.append(p)
        diffs.append(min_vals.mean() - freq_vals.mean())
        names.append(m)
        min_eces.append(min_vals.mean())
        freq_eces.append(freq_vals.mean())
    adj = holm_correct(pvals)
    with open("results_v2/table_h8_raw_ece_paired_ttest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "minority_ece_mean", "frequent_ece_mean", "ratio_minority_over_frequent",
                    "diff_minority_minus_frequent", "p_one_sided", "p_holm_6models", "sig_holm_0.05"])
        for m, minE, freqE, diff, p, padj in zip(names, min_eces, freq_eces, diffs, pvals, adj):
            ratio = minE / freqE
            w.writerow([m, round(minE, 4), round(freqE, 4), round(ratio, 4),
                        round(diff, 4), round(p, 6), round(padj, 4), padj < 0.05])
    print("Saved results_v2/table_h8_raw_ece_paired_ttest.csv")


if __name__ == "__main__":
    main()
