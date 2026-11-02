"""
Recompute headline metrics on the correctly-screened 61,701-comment test set (both the
near-duplicate screen AND the short-comment screen combined), vs. the 63,662-comment set
(short-comment screen only) that every saved result in this project actually used.

Background: pre_registration.md and the Paper-3 manuscript describe a combined screen yielding
61,701 test comments, but src/duplicate_audit.py's near-duplicate flags (data/duplicate_screen.npz)
were computed and saved, never applied as a filter anywhere in training, prediction or
generate_results.py. Every saved results_proposed/runs/<model>_p4/seed_<seed>/test_outputs.npz has
63,662 rows (= 63,978 - the 316-comment short-comment screen only). This script answers: does any
headline conclusion change once the near-duplicate comments are also excluded?

No retraining, no new predictions: re-aggregates the already-saved test_outputs.npz arrays after
masking out near-duplicate rows. The row alignment between test_outputs.npz (63,662 rows) and
duplicate_screen.npz (63,978 rows, indexed by comment id) is reconstructed deterministically by
replaying the exact same loading/filtering steps as src/data.py's get_splits(min_words=3), which
involves no shuffling (pd.read_csv -> merge -> dropna -> boolean-mask filtering, all order-
preserving) and does not depend on seed for the test set.

Usage: python -m src.recompute_on_correctly_screened_test
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data import load_official_test, LABELS

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"
MINORITY = ["threat", "severe_toxic", "identity_hate"]
FREQUENT = ["toxic", "obscene", "insult"]


def reconstruct_test_id_order(data_dir="data", min_words=3):
    """Exactly replays get_splits' test-set loading/filtering, order-preserving, to recover the
    id sequence that matches test_outputs.npz's row order (63,662 rows)."""
    test_df = load_official_test(data_dir)
    mask = test_df["comment_text"].str.split().str.len() >= min_words
    test_df = test_df[mask].reset_index(drop=True)
    return test_df["id"].values


def build_near_dup_mask(test_ids_63662):
    d = np.load("data/duplicate_screen.npz", allow_pickle=True)
    id_to_dup = dict(zip(d["test_id"], d["is_near_duplicate"]))
    missing = [i for i in test_ids_63662[:5] if i not in id_to_dup]
    if missing:
        raise RuntimeError(f"ID alignment failed, sample missing ids: {missing}")
    return np.array([bool(id_to_dup[i]) for i in test_ids_63662])


def macro_f1_m(y, p, threshold=0.5):
    y_pred = (p >= threshold).astype(int)
    precs = [precision_score(y[:, j], y_pred[:, j], zero_division=0) for j in range(y.shape[1])]
    recs = [recall_score(y[:, j], y_pred[:, j], zero_division=0) for j in range(y.shape[1])]
    P, R = np.mean(precs), np.mean(recs)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def macro_auc(y, p):
    aucs = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0 or y[:, j].sum() == len(y):
            continue
        aucs.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(aucs))


def ece_label(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = len(p)
    return float(sum(abs(y[idx == m].mean() - p[idx == m].mean()) * (idx == m).mean()
                      for m in range(bins) if (idx == m).any()))


def macro_ece(y, p):
    return float(np.mean([ece_label(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def load_thresholds(model, seed):
    import json
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    thr = m.get("thresholds")
    if thr is None:
        return 0.5
    if isinstance(thr, dict):
        return np.array([thr[label] for label in LABELS])
    return np.array(thr)


def main():
    print("=" * 70)
    print("RECOMPUTE ON CORRECTLY-SCREENED TEST SET (near-dup + short-comment, 61,701 comments)")
    print("vs. the 63,662-comment set every saved result in this project actually used")
    print("=" * 70)

    test_ids = reconstruct_test_id_order()
    print(f"\nReconstructed test id order: {len(test_ids)} rows (expect 63,662)")
    dup_mask = build_near_dup_mask(test_ids)
    print(f"Near-duplicate rows within the 63,662 set: {int(dup_mask.sum())} "
          f"(expect 1,961 = 2,029 total near-dups - 68 already excluded by the short-comment screen)")
    keep_mask = ~dup_mask
    print(f"Correctly-screened row count: {int(keep_mask.sum())} (expect 61,701)")

    rows = []
    for m in MODELS:
        for s in SEEDS:
            d = np.load(os.path.join(RESULTS_DIR, f"{m}_p4", f"seed_{s}", "test_outputs.npz"))
            probs, y = d["probs"], d["y_true"]
            assert len(probs) == len(test_ids), f"{m} seed {s}: row mismatch {len(probs)} vs {len(test_ids)}"
            thr = load_thresholds(m, s)

            f1m_before = macro_f1_m(y, probs, thr)
            auc_before = macro_auc(y, probs)
            ece_before = macro_ece(y, probs)

            probs_s, y_s = probs[keep_mask], y[keep_mask]
            f1m_after = macro_f1_m(y_s, probs_s, thr)
            auc_after = macro_auc(y_s, probs_s)
            ece_after = macro_ece(y_s, probs_s)

            rows.append({
                "model": m, "seed": s,
                "f1_macro_m_63662": f1m_before, "f1_macro_m_61701": f1m_after,
                "roc_auc_macro_63662": auc_before, "roc_auc_macro_61701": auc_after,
                "ece_macro_63662": ece_before, "ece_macro_61701": ece_after,
            })
    df = pd.DataFrame(rows)
    df.round(6).to_csv("results_proposed/report/table_rescreened_61701_per_seed.csv", index=False)

    print("\n=== Per-model mean over 3 seeds: before (63,662) vs after (61,701) ===")
    summary_rows = []
    for m in MODELS:
        sub = df[df["model"] == m]
        row = {"model": m}
        for metric in ["f1_macro_m", "roc_auc_macro", "ece_macro"]:
            b = sub[f"{metric}_63662"].mean()
            a = sub[f"{metric}_61701"].mean()
            row[f"{metric}_63662"] = b
            row[f"{metric}_61701"] = a
            row[f"{metric}_diff"] = a - b
        summary_rows.append(row)
        print(f"  {m:12s} F1(M) {row['f1_macro_m_63662']:.4f}->{row['f1_macro_m_61701']:.4f} "
              f"({row['f1_macro_m_diff']:+.4f})  "
              f"AUC {row['roc_auc_macro_63662']:.4f}->{row['roc_auc_macro_61701']:.4f} "
              f"({row['roc_auc_macro_diff']:+.4f})  "
              f"ECE {row['ece_macro_63662']:.4f}->{row['ece_macro_61701']:.4f} "
              f"({row['ece_macro_diff']:+.4f})")
    summary = pd.DataFrame(summary_rows)
    summary.round(6).to_csv("results_proposed/report/table_rescreened_61701_summary.csv", index=False)

    print("\n=== H3a/H3b retest: macro-F1(M) vs Maslej-Kresnakova benchmark, on 61,701 set ===")
    MASLEJ = {"lstm": 0.56, "bilstm": 0.56, "gru": 0.56, "bert": 0.55, "distilbert": 0.55, "roberta": 0.55}
    h3_rows = []
    pvals = []
    for m in MODELS:
        sub = df[df["model"] == m]
        vals = sub["f1_macro_m_61701"].values
        diff = vals.mean() - MASLEJ[m]
        t, p_two = stats.ttest_1samp(vals, MASLEJ[m])
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        pvals.append(p_one)
        h3_rows.append({"model": m, "mean_f1_macro_m_61701": vals.mean(), "benchmark": MASLEJ[m],
                        "diff": diff, "seeds_above": int((vals > MASLEJ[m]).sum()), "p_one_sided": p_one})
    h3 = pd.DataFrame(h3_rows)
    order = np.argsort(pvals)
    adj = [0.0] * 6
    running = 0.0
    for rank, i in enumerate(order):
        c = pvals[i] * (6 - rank)
        running = max(running, c)
        adj[i] = min(running, 1.0)
    h3["p_holm_6models"] = adj
    h3["sig_0.05"] = h3["p_holm_6models"] < 0.05
    print(h3.to_string(index=False))
    h3.round(6).to_csv("results_proposed/report/table_rescreened_h3_retest.csv", index=False)

    print(f"\nFull per-seed table: results_proposed/report/table_rescreened_61701_per_seed.csv")
    print(f"Summary table: results_proposed/report/table_rescreened_61701_summary.csv")
    print(f"H3a/H3b retest: results_proposed/report/table_rescreened_h3_retest.csv")


if __name__ == "__main__":
    main()
