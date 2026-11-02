"""
Seed-stability (coefficient of variation) and threshold-tuning gain, across all six models.

Compares CV(F1_macro_M), CV(ROC-AUC macro) against CV(ECE), CV(Brier) -- tests whether
calibration metrics are more seed-sensitive than discriminative metrics (relevant to any
single-run study's inability to have caught a calibration result reliably).

Also reports threshold-tuning gain (0.5 -> validation-tuned) and training time per model.

No new training. Reads results_proposed/runs/<model>_p{3,4}/seed_<seed>/metrics.json only.

Usage: python -m src.seed_stability
"""
import json
import os
import numpy as np

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"


def load(model, stage, seed):
    path = os.path.join(RESULTS_DIR, f"{model}_{stage}", f"seed_{seed}", "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test"]


def cv(arr):
    arr = np.array(arr)
    if len(arr) < 2:
        return float("nan")
    return arr.std(ddof=1) / arr.mean()


def main():
    print("=" * 70)
    print("SEED STABILITY + THRESHOLD GAIN + TRAINING TIME -- all six models")
    print("=" * 70)

    print("\n=== Threshold-tuning gain: F1_macro(M) at 0.5 vs tuned ===")
    for m in MODELS:
        f1_05, f1_tuned = [], []
        for s in SEEDS:
            d3, d4 = load(m, "p3", s), load(m, "p4", s)
            if d3 is None or d4 is None:
                continue
            f1_05.append(d3["f1_macro_m"])
            f1_tuned.append(d4["f1_macro_m"])
        if not f1_05:
            print(f"  {m:12s} NO DATA")
            continue
        f1_05, f1_tuned = np.array(f1_05), np.array(f1_tuned)
        gain = f1_tuned - f1_05
        n_note = "" if len(f1_05) == 3 else f" (n={len(f1_05)}, partial)"
        print(f"  {m:12s} @0.5={f1_05.mean():.4f}  tuned={f1_tuned.mean():.4f}  "
              f"gain={gain.mean():+.4f} (+{100*gain.mean()/f1_05.mean():.1f}%){n_note}")

    print("\n=== Seed stability: CV of discriminative metrics (F1_macro_M, ROC-AUC), tuned thresholds ===")
    cv_f1, cv_auc, cv_ece, cv_brier = {}, {}, {}, {}
    for m in MODELS:
        f1s, aucs, eces, briers = [], [], [], []
        for s in SEEDS:
            d = load(m, "p4", s)
            if d is None:
                continue
            f1s.append(d["f1_macro_m"])
            aucs.append(d["roc_auc_macro"])
            eces.append(d["ece_macro"])
            briers.append(d["brier_macro"])
        if not f1s:
            continue
        cv_f1[m], cv_auc[m], cv_ece[m], cv_brier[m] = cv(f1s), cv(aucs), cv(eces), cv(briers)
        n_note = "" if len(f1s) == 3 else f" (n={len(f1s)}, partial)"
        print(f"  {m:12s} CV(F1_M)={cv_f1[m]*100:.2f}%  CV(AUC)={cv_auc[m]*100:.3f}%{n_note}")

    print("\n=== Seed stability: CV of calibration metrics (ECE, Brier) -- NEW ===")
    for m in MODELS:
        if m not in cv_ece:
            continue
        n_seeds = sum(1 for s in SEEDS if load(m, "p4", s) is not None)
        n_note = "" if n_seeds == 3 else f" (n={n_seeds}, partial)"
        print(f"  {m:12s} CV(ECE)={cv_ece[m]*100:.2f}%  CV(Brier)={cv_brier[m]*100:.2f}%{n_note}")

    print("\n=== Is calibration MORE seed-sensitive than discrimination? (CV comparison, H4-style) ===")
    for m in MODELS:
        if m not in cv_ece or np.isnan(cv_ece[m]) or np.isnan(cv_f1[m]):
            continue
        more_sensitive_ece = "YES" if cv_ece[m] > cv_f1[m] else "no"
        more_sensitive_brier = "YES" if cv_brier[m] > cv_auc[m] else "no"
        print(f"  {m:12s} CV(ECE) > CV(F1_M)? {more_sensitive_ece}   CV(Brier) > CV(AUC)? {more_sensitive_brier}")

    print("\n=== Training time per run (minutes), mean over available seeds ===")
    # BERT seed 42 ran concurrently with another BERT job and was GPU-contention-starved
    # (284.85 min vs. 65.59/86.81 min for seeds 43/44, run solo) -- a wall-clock measurement
    # artifact, not a model-quality difference. Per user instruction (1 Nov 2026), reporting
    # substitutes the mean of seeds 43/44 for seed 42. Raw metrics.json is left untouched.
    for m in MODELS:
        times_by_seed = {}
        for s in SEEDS:
            d = load(m, "p3", s)
            if d is not None:
                path = os.path.join(RESULTS_DIR, f"{m}_p3", f"seed_{s}", "metrics.json")
                with open(path, encoding="utf-8") as f:
                    full = json.load(f)
                if "train_minutes" in full:
                    times_by_seed[s] = full["train_minutes"]
        note = ""
        if m == "bert" and 42 in times_by_seed and 43 in times_by_seed and 44 in times_by_seed:
            raw_42 = times_by_seed[42]
            times_by_seed[42] = (times_by_seed[43] + times_by_seed[44]) / 2
            note = f"  [seed 42 corrected: raw {raw_42:.1f} min was GPU-contention-starved, " \
                   f"substituted mean(seed43,seed44)={times_by_seed[42]:.1f} min]"
        times = list(times_by_seed.values())
        if times:
            n_note = "" if len(times) == 3 else f" (n={len(times)}, partial)"
            print(f"  {m:12s} mean={np.mean(times):.1f} min  values={[round(t,1) for t in times]}{n_note}{note}")


if __name__ == "__main__":
    main()
