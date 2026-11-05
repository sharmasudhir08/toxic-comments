"""
Paper-3 Stage 1.3: H1 family-level comparison, PRIMARY test, per pre_registration.md's own H1 row
(nested ANOVA F(1,4), models nested in families) -- not Paper-3's own family_comparison.py, which
only ever computed Welch's t-test (kept here as the secondary/robustness report, matching
pre_registration.md's own structure).

Primary analysis: duplicate-screened test set (61,701 of 63,662 -- near-duplicates of a training
comment, TF-IDF cosine >= 0.9, removed from metric computation only; same predictions used).
Sensitivity: full 63,662-comment test set (no screening).

Reads only saved predictions (test_outputs.npz) and saved tuned thresholds (metrics.json). No
retraining, no new inference -- metrics are recomputed from already-saved probabilities on a
boolean subset of rows.

Usage: python -m src.h1_nested_anova_paper3   (run from E:\\Toxic Comments)
"""
import json
import os

import numpy as np
import pandas as pd

from src.metrics import multilabel_metrics
from src.config import LABELS

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
MODELS = RECURRENT + TRANSFORMER
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"
METRICS = ["f1_macro_m", "roc_auc_macro", "pr_auc_macro", "ece_macro"]


def load_run(model, seed):
    outputs = np.load(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz"))
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        meta = json.load(f)
    thresholds = np.array([meta["thresholds"][lab] for lab in LABELS])
    return outputs["probs"], outputs["y_true"], thresholds


def compute_metrics(probs, y_true, thresholds, mask=None):
    if mask is not None:
        probs = probs[mask]
        y_true = y_true[mask]
    m = multilabel_metrics(y_true, probs, thresholds)
    return {
        "f1_macro_m": m["f1_macro_m"],
        "roc_auc_macro": m["roc_auc_macro"],
        "pr_auc_macro": m["pr_auc_macro"],
        "ece_macro": m["ece_macro"],
    }


def nested_anova(long_df, metric):
    d = long_df[["family", "model", "value"]].copy()
    grand_mean = d["value"].mean()
    n_total = len(d)
    fam_means = d.groupby("family")["value"].mean()
    model_means = d.groupby(["family", "model"])["value"].mean()
    n_seeds_per_model = d.groupby("model").size().iloc[0]
    n_models_per_family = d.groupby("family")["model"].nunique()
    n_families = fam_means.shape[0]
    n_models = model_means.shape[0]

    ss_family = sum(n_models_per_family[fam] * n_seeds_per_model * (fmean - grand_mean) ** 2
                     for fam, fmean in fam_means.items())
    df_family = n_families - 1
    ss_model_in_family = sum(n_seeds_per_model * (mmean - fam_means[fam]) ** 2
                              for (fam, model), mmean in model_means.items())
    df_model_in_family = n_models - n_families
    ss_error = sum(((grp["value"] - grp["value"].mean()) ** 2).sum() for _, grp in d.groupby("model"))
    df_error = n_total - n_models

    ms_family = ss_family / df_family
    ms_model_in_family = ss_model_in_family / df_model_in_family
    from scipy import stats
    f_stat = ms_family / ms_model_in_family
    p_val = stats.f.sf(f_stat, df_family, df_model_in_family)
    return {
        "metric": metric, "ss_family": ss_family, "df_family": df_family,
        "ss_model_in_family": ss_model_in_family, "df_model_in_family": df_model_in_family,
        "ms_family": ms_family, "ms_model_in_family": ms_model_in_family,
        "F_family": f_stat, "p_family": p_val,
        "family_mean_recurrent": fam_means.get("recurrent"),
        "family_mean_transformer": fam_means.get("transformer"),
    }


def main():
    dup_mask = np.load("data/primary_mask_63662.npy")  # True = near-duplicate (exclude from primary)
    keep_mask = ~dup_mask
    print(f"Duplicate-screened primary set: {keep_mask.sum()} of {len(keep_mask)} comments kept.")

    rows_primary, rows_sensitivity = [], []
    for model in MODELS:
        family = "recurrent" if model in RECURRENT else "transformer"
        for seed in SEEDS:
            probs, y_true, thresholds = load_run(model, seed)
            assert len(probs) == len(keep_mask), f"{model} seed {seed}: length mismatch"
            m_primary = compute_metrics(probs, y_true, thresholds, mask=keep_mask)
            m_sens = compute_metrics(probs, y_true, thresholds, mask=None)
            for metric in METRICS:
                rows_primary.append({"family": family, "model": model, "seed": seed,
                                      "metric": metric, "value": m_primary[metric]})
                rows_sensitivity.append({"family": family, "model": model, "seed": seed,
                                          "metric": metric, "value": m_sens[metric]})

    df_primary = pd.DataFrame(rows_primary)
    df_sens = pd.DataFrame(rows_sensitivity)

    print("\n=== H1 PRIMARY: nested ANOVA F(1,4), duplicate-screened (61,701) ===")
    results_primary = []
    for metric in METRICS:
        sub = df_primary[df_primary["metric"] == metric]
        r = nested_anova(sub, metric)
        results_primary.append(r)
        print(f"  {metric:15s} F(1,4)={r['F_family']:.5f}  p={r['p_family']:.5f}  "
              f"recurrent={r['family_mean_recurrent']:.4f}  transformer={r['family_mean_transformer']:.4f}")

    print("\n=== H1 SENSITIVITY: nested ANOVA F(1,4), full unscreened (63,662) ===")
    results_sens = []
    for metric in METRICS:
        sub = df_sens[df_sens["metric"] == metric]
        r = nested_anova(sub, metric)
        results_sens.append(r)
        print(f"  {metric:15s} F(1,4)={r['F_family']:.5f}  p={r['p_family']:.5f}  "
              f"recurrent={r['family_mean_recurrent']:.4f}  transformer={r['family_mean_transformer']:.4f}")

    os.makedirs("results_v2", exist_ok=True)
    pd.DataFrame(results_primary).round(6).to_csv("results_v2/table_h1_nested_anova_primary_screened.csv", index=False)
    pd.DataFrame(results_sens).round(6).to_csv("results_v2/table_h1_nested_anova_sensitivity_full.csv", index=False)
    df_primary.round(6).to_csv("results_v2/table_h1_long_primary_screened.csv", index=False)
    df_sens.round(6).to_csv("results_v2/table_h1_long_sensitivity_full.csv", index=False)
    print("\nWrote results_v2/table_h1_nested_anova_primary_screened.csv")
    print("Wrote results_v2/table_h1_nested_anova_sensitivity_full.csv")


if __name__ == "__main__":
    main()
