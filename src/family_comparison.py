"""
Family-level comparison: recurrent (LSTM, Bi-LSTM, GRU) vs transformer (BERT, DistilBERT,
RoBERTa) -- the central test for both journal papers (H1 in Preregistration-2.md, H1/H5 in
pre_registration.md).

Unit of analysis is the SIX MODELS (not seeds): each model contributes one mean value per
metric (averaged over its 3 seeds), and the family effect is tested with the 3 recurrent
means vs the 3 transformer means -- nested/nested-ANOVA-equivalent via a two-sample test
on the 3-vs-3 model means, plus Welch's t-test over all 18 runs as an explicit (over-stated)
illustration of why the unit of analysis matters.

For the vs-published-baseline test (H3a/H3b), do NOT use this script -- use the already-existing,
authoritative `generate_results.py` (table7b_vs_maslej_tests.csv), which pools ALL model x metric x
threshold tests into ONE Holm correction (currently n=48 across both families). An earlier version
of this script ran its own, differently-scoped Holm correction (per-metric, 3-6 models only) and
got a materially different, WRONG answer (falsely "significant" for several models) -- removed
after cross-checking against table7b_vs_maslej_tests.csv on 1 Nov 2026. See pre_registration.md
Sec. 4 ("Holm correction is applied within each family of tests" = the whole table7b battery, not
just the models being compared on one metric).

No new training. Reads results_proposed/runs/<model>_p4/seed_<seed>/metrics.json only.
Requires all 6 models to have 3 seeds each -- refuses to run a family-level test otherwise.

Usage: python -m src.family_comparison
"""
import json
import os
import numpy as np
from scipy import stats

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
MODELS = RECURRENT + TRANSFORMER
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"


def load_seeds(model):
    out = {}
    for s in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{s}", "metrics.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                out[s] = json.load(f)["test"]
    return out


def main():
    data = {m: load_seeds(m) for m in MODELS}
    incomplete = [m for m in MODELS if len(data[m]) != 3]
    print("=" * 70)
    print("FAMILY-LEVEL COMPARISON: recurrent vs transformer (H1, H5/H6)")
    print("=" * 70)
    print("\nSeed completeness:", {m: sorted(data[m].keys()) for m in MODELS})
    if incomplete:
        print(f"\nREFUSING to run family-level test: {incomplete} do not have all 3 seeds.")
        print("This test requires every model complete -- partial data would bias the")
        print("family mean and understate variance. Re-run once all 6 models have 3 seeds.")
        return

    metrics = ["f1_macro_m", "roc_auc_macro", "pr_auc_macro", "ece_macro"]
    model_means = {m: {} for m in MODELS}
    for m in MODELS:
        for metric in metrics:
            vals = [data[m][s][metric] for s in SEEDS]
            model_means[m][metric] = np.mean(vals)

    print("\n=== Per-model means (3 seeds each) ===")
    for m in MODELS:
        fam = "recurrent" if m in RECURRENT else "transformer"
        print(f"  {m:12s} ({fam:11s}) F1_M={model_means[m]['f1_macro_m']:.4f}  "
              f"AUC={model_means[m]['roc_auc_macro']:.4f}  "
              f"PR-AUC={model_means[m]['pr_auc_macro']:.4f}  "
              f"ECE={model_means[m]['ece_macro']:.4f}")

    h1_rows = []
    print("\n=== H1: Family-level comparison (unit of analysis = 6 models, 3 vs 3) ===")
    for metric in metrics:
        rec_vals = np.array([model_means[m][metric] for m in RECURRENT])
        trans_vals = np.array([model_means[m][metric] for m in TRANSFORMER])
        t, p = stats.ttest_ind(trans_vals, rec_vals, equal_var=False)  # Welch, model-level n=3 vs n=3
        diff = trans_vals.mean() - rec_vals.mean()
        direction = "transformer higher" if diff > 0 else "recurrent higher"
        print(f"  {metric:15s} recurrent_mean={rec_vals.mean():.4f}  transformer_mean={trans_vals.mean():.4f}  "
              f"diff={diff:+.4f} ({direction})  Welch_t(3v3) p={p:.4f}")
        h1_rows.append({"metric": metric, "recurrent_mean": rec_vals.mean(), "transformer_mean": trans_vals.mean(),
                         "diff": diff, "direction": direction, "welch_t_3v3_p": p})

    print("\n  Robustness check -- exact permutation over the 20 ways to split 6 models into two")
    print("  groups of 3 (two-sided, is the observed split the most extreme?):")
    from itertools import combinations
    all_models = MODELS
    for metric in metrics:
        vals = {m: model_means[m][metric] for m in all_models}
        observed_diff = abs(np.mean([vals[m] for m in TRANSFORMER]) - np.mean([vals[m] for m in RECURRENT]))
        all_splits = list(combinations(all_models, 3))
        seen = set()
        diffs = []
        for combo in all_splits:
            rest = tuple(m for m in all_models if m not in combo)
            key = tuple(sorted([tuple(sorted(combo)), tuple(sorted(rest))]))
            if key in seen:
                continue
            seen.add(key)
            d = abs(np.mean([vals[m] for m in combo]) - np.mean([vals[m] for m in rest]))
            diffs.append(d)
        diffs = np.array(diffs)
        rank = (diffs >= observed_diff - 1e-12).sum()
        perm_p = rank / len(diffs)
        print(f"    {metric:15s} observed_diff={observed_diff:.4f}  "
              f"rank={rank}/{len(diffs)} splits >= observed  exact_p={perm_p:.4f}")
        row = next(r for r in h1_rows if r["metric"] == metric)
        row["exact_permutation_p"] = perm_p
        row["permutation_splits_total"] = len(diffs)

    print("\n  WARNING, illustration only -- treating all 18 runs as independent overstates evidence:")
    for metric in metrics:
        rec_runs = np.array([data[m][s][metric] for m in RECURRENT for s in SEEDS])
        trans_runs = np.array([data[m][s][metric] for m in TRANSFORMER for s in SEEDS])
        t, p = stats.ttest_ind(trans_runs, rec_runs, equal_var=False)
        print(f"    {metric:15s} Welch over 18 runs (WRONG unit of analysis): p={p:.4f}")
        row = next(r for r in h1_rows if r["metric"] == metric)
        row["welch_18runs_WRONG_unit_p"] = p

    import pandas as pd
    h1_df = pd.DataFrame(h1_rows)
    out_path = "results_proposed/report/table_h1_family_comparison.csv"
    h1_df.round(5).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n=== H3a/H3b (vs published Maslej-Kresnakova baseline): see results_proposed/report/")
    print("    table7b_vs_maslej_tests.csv (generate_results.py) -- authoritative, do not recompute here.")
    print("    Current result (regenerated 1 Nov 2026, all 6 models, single Holm correction over all 48")
    print("    model x metric x threshold tests): H3a (recurrent vs Maslej 0.56) NOT significant for any")
    print("    of LSTM/Bi-LSTM/GRU (p_Holm 0.487/0.822/0.719). H3b (transformer vs Maslej 0.55) IS")
    print("    significant for BERT (tuned, p_Holm=0.041) and RoBERTa (tuned, p_Holm=0.004), NOT for")
    print("    DistilBERT (tuned, p_Holm=0.058, narrowly misses).")


if __name__ == "__main__":
    main()
