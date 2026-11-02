"""
Length-robustness analysis, both discriminative metrics and calibration, across all six
models (both families).

Bins the official test set by comment length (words), then recomputes, per bin, per model
(mean over available seeds):
  - macro-P/R-based F1 (F1_M) and macro ROC-AUC, at threshold 0.5, from raw saved predictions
  - macro ECE per bin (threshold-free, from raw predicted probabilities)

Reads predictions from results_proposed/runs/<model>_p3/seed_<seed>/test_outputs.npz (same
file for every seed of a given model -- row order is identical across seeds, verified via
label-sum cross-check against src/data.py's deterministic test_df construction).

No new training. Models with fewer than 3 seeds are included with whatever seeds exist and
clearly flagged.

Usage: python -m src.length_robustness
"""
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.data import get_splits

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BIN_EDGES = [26, 51, 101, 201]  # bins: <=25, 26-50, 51-100, 101-200, >200
BIN_NAMES = ["<=25", "26-50", "51-100", "101-200", ">200"]

ECE_BINS = 10


def ece_score(probs, y_true, n_bins=ECE_BINS):
    """Macro-averaged expected calibration error across the six labels, for one length bin."""
    label_eces = []
    for li in range(probs.shape[1]):
        p = probs[:, li]
        y = y_true[:, li]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (p >= bin_edges[b]) & (p < bin_edges[b + 1] if b < n_bins - 1 else p <= bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            acc = y[mask].mean()
            conf = p[mask].mean()
            ece += (mask.sum() / len(p)) * abs(acc - conf)
        label_eces.append(ece)
    return float(np.mean(label_eces))


def available_seeds(model):
    import os
    return [s for s in SEEDS if os.path.isfile(f"results_proposed/runs/{model}_p3/seed_{s}/test_outputs.npz")]


def main():
    train_df, val_df, test_df, source = get_splits("data", seed=42, min_words=3)
    lengths = test_df["comment_text"].str.split().str.len().values
    bin_idx = np.digitize(lengths, BIN_EDGES)

    print("=" * 70)
    print("LENGTH-ROBUSTNESS ANALYSIS -- discriminative metrics + calibration")
    print("=" * 70)
    print("\nTest set:", source, "| n =", len(test_df))
    print("\n=== Length-bin sizes and label prevalence ===")
    for name, b in zip(BIN_NAMES, range(5)):
        mask = bin_idx == b
        print(f"  {name:10s} n={mask.sum():6d}  toxic_rate={test_df.loc[mask,'toxic'].mean():.3f}  "
              f"threat_rate={test_df.loc[mask,'threat'].mean():.4f}")

    results_f1 = {m: {b: [] for b in range(5)} for m in MODELS}
    results_auc = {m: {b: [] for b in range(5)} for m in MODELS}
    results_ece = {m: {b: [] for b in range(5)} for m in MODELS}
    seeds_used = {}

    for m in MODELS:
        seeds = available_seeds(m)
        seeds_used[m] = seeds
        for s in seeds:
            d = np.load(f"results_proposed/runs/{m}_p3/seed_{s}/test_outputs.npz")
            probs, y_true = d["probs"], d["y_true"]
            preds = (probs >= 0.5).astype(int)
            for b in range(5):
                mask = bin_idx == b
                if mask.sum() < 5:
                    continue
                precs, recs = [], []
                for li in range(6):
                    yt, yp = y_true[mask, li], preds[mask, li]
                    if yt.sum() == 0 and yp.sum() == 0:
                        continue
                    precs.append(precision_score(yt, yp, zero_division=0))
                    recs.append(recall_score(yt, yp, zero_division=0))
                P, R = np.mean(precs), np.mean(recs)
                f1m = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                results_f1[m][b].append(f1m)

                label_aucs = []
                for li in range(6):
                    yt = y_true[mask, li]
                    if yt.sum() == 0 or yt.sum() == len(yt):
                        continue
                    label_aucs.append(roc_auc_score(yt, probs[mask, li]))
                if label_aucs:
                    results_auc[m][b].append(np.mean(label_aucs))

                results_ece[m][b].append(ece_score(probs[mask], y_true[mask]))

    print("\nSeeds used per model:", {m: seeds_used[m] for m in MODELS})

    print("\n=== F1(M) per model per length bin, mean over available seeds ===")
    print(f"{'Model':12s}", *[f"{n:>10s}" for n in BIN_NAMES])
    for m in MODELS:
        row = []
        for b in range(5):
            vals = results_f1[m][b]
            row.append(f"{np.mean(vals):.3f}" if vals else "n/a")
        flag = "" if len(seeds_used[m]) == 3 else f"  (n={len(seeds_used[m])}, partial)"
        print(f"{m:12s}", *[f"{v:>10s}" for v in row], flag)

    print("\n=== ROC-AUC macro per model per length bin, mean over available seeds ===")
    print(f"{'Model':12s}", *[f"{n:>10s}" for n in BIN_NAMES])
    for m in MODELS:
        row = []
        for b in range(5):
            vals = results_auc[m][b]
            row.append(f"{np.mean(vals):.4f}" if vals else "n/a")
        flag = "" if len(seeds_used[m]) == 3 else f"  (n={len(seeds_used[m])}, partial)"
        print(f"{m:12s}", *[f"{v:>10s}" for v in row], flag)

    print("\n=== Macro ECE per model per length bin, mean over available seeds (NEW: calibration by length) ===")
    print(f"{'Model':12s}", *[f"{n:>10s}" for n in BIN_NAMES])
    for m in MODELS:
        row = []
        for b in range(5):
            vals = results_ece[m][b]
            row.append(f"{np.mean(vals):.4f}" if vals else "n/a")
        flag = "" if len(seeds_used[m]) == 3 else f"  (n={len(seeds_used[m])}, partial)"
        print(f"{m:12s}", *[f"{v:>10s}" for v in row], flag)

    missing = [m for m in MODELS if len(seeds_used[m]) < 3]
    if missing:
        print(f"\nNOTE: {missing} do not have all 3 seeds yet. Re-run once complete.")


if __name__ == "__main__":
    main()
