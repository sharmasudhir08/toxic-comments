"""
Paper-3 Stage 1.3: H3a/H3b and H4, threshold-matched against Maslej et al. (2020).

Fixes the threshold mismatch confirmed in src/family_comparison.py's own authoritative
table7b_vs_maslej_tests.csv and in Paper-3-DC.docm's own Table 3/4/A3: the 0.5-threshold (P3)
rows must be compared against Maslej's UNTUNED values (0.56 recurrent, 0.55 transformer); only
the tuned (P4) rows go against Maslej's TUNED values (0.592 recurrent, 0.621 transformer,
mechanically recomputed from their own published per-class P/R, Tables 17-18 -- see reference.md
Sec. A2.1). One consistent sign convention throughout: diff = ours - benchmark.

Test set note: Paper-3 evaluates on 63,662 comments (316 short comments removed before
training/prediction); Maslej's own study used the full 63,978. This is disclosed, not corrected
for, since Maslej's own comparator values are fixed regardless of our test-set size.

No new training. Reads results_proposed/runs/<model>_p3|p4/seed_<seed>/metrics.json only.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 43, 44]
MODELS = {
    "lstm": ("LSTM", "recurrent"), "bilstm": ("Bi-LSTM", "recurrent"), "gru": ("GRU", "recurrent"),
    "bert": ("BERT", "transformer"), "distilbert": ("DistilBERT", "transformer"),
    "roberta": ("RoBERTa", "transformer"),
}
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BENCHMARK_UNTUNED = {"recurrent": 0.56, "transformer": 0.55}
BENCHMARK_TUNED = {"recurrent": 0.592, "transformer": 0.621}
MASLEJ_PER_LABEL_UNTUNED = {
    "recurrent": dict(zip(LABELS, [0.68, 0.41, 0.69, 0.34, 0.68, 0.54])),
    "transformer": dict(zip(LABELS, [0.70, 0.29, 0.69, 0.47, 0.69, 0.60])),
}
MASLEJ_PER_LABEL_TUNED = {
    "recurrent": dict(zip(LABELS, [0.70, 0.44, 0.69, 0.34, 0.68, 0.55])),
    "transformer": dict(zip(LABELS, [0.71, 0.41, 0.71, 0.47, 0.69, 0.65])),
}
MINORITY = ["threat", "severe_toxic", "identity_hate"]
FREQUENT = ["toxic", "obscene", "insult"]


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


def load_f1_macro_m(model, stage, seed):
    path = f"results_proposed/runs/{model}_{stage}/seed_{seed}/metrics.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test"]["f1_macro_m"]


def load_per_label_f1(model, stage, seed):
    path = f"results_proposed/runs/{model}_{stage}/seed_{seed}/metrics.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {lab: d["test"]["per_label"][lab]["f1"] for lab in LABELS}


def h3_rows(stage, benchmark):
    rows = []
    for key, (label, family) in MODELS.items():
        vals = np.array([load_f1_macro_m(key, stage, s) for s in SEEDS])
        target = benchmark[family]
        diff = vals.mean() - target
        t_stat, p_two = stats.ttest_1samp(vals, target)
        p_one = p_two / 2 if diff > 0 else 1 - p_two / 2
        rows.append({
            "model": label, "family": family, "threshold_stage": stage,
            "mean_f1_macro_m": vals.mean(), "benchmark": target, "diff": diff,
            "seeds_above": f"{(vals > target).sum()}/3", "t_stat": t_stat, "df": 2,
            "p_one_sided": p_one,
        })
    df = pd.DataFrame(rows)
    df["p_holm_6models"] = holm(df["p_one_sided"].tolist())
    df["sig_0.05"] = df["p_holm_6models"] < 0.05
    return df


def h4_rows(stage, maslej_per_label):
    rows = []
    for key, (label, family) in MODELS.items():
        per_label_means = {lab: np.mean([load_per_label_f1(key, stage, s)[lab] for s in SEEDS])
                            for lab in LABELS}
        minority_margin = np.mean([per_label_means[lab] - maslej_per_label[family][lab] for lab in MINORITY])
        frequent_margin = np.mean([per_label_means[lab] - maslej_per_label[family][lab] for lab in FREQUENT])
        rows.append({"model": label, "family": family, "threshold_stage": stage,
                      "mean_minority_margin": minority_margin, "mean_frequent_margin": frequent_margin})
    return pd.DataFrame(rows)


def main():
    os.makedirs("results_v2", exist_ok=True)

    print("=" * 70)
    print("H3a/H3b: threshold-matched vs. Maslej et al. (2020)")
    print("=" * 70)
    h3_untuned = h3_rows("p3", BENCHMARK_UNTUNED)
    print("\n--- 0.5-threshold (p3) vs. UNTUNED Maslej (0.56 recurrent / 0.55 transformer) ---")
    print(h3_untuned.round(5).to_string(index=False))

    h3_tuned = h3_rows("p4", BENCHMARK_TUNED)
    print("\n--- Tuned threshold (p4) vs. TUNED Maslej (0.592 recurrent / 0.621 transformer) ---")
    print(h3_tuned.round(5).to_string(index=False))

    h3_untuned.round(5).to_csv("results_v2/table_h3_vs_maslej_untuned.csv", index=False)
    h3_tuned.round(5).to_csv("results_v2/table_h3_vs_maslej_tuned.csv", index=False)

    print("\n" + "=" * 70)
    print("H4: minority vs. frequent margin over Maslej's per-label values")
    print("=" * 70)
    h4_untuned = h4_rows("p3", MASLEJ_PER_LABEL_UNTUNED)
    h4_tuned = h4_rows("p4", MASLEJ_PER_LABEL_TUNED)
    print("\n--- 0.5-threshold (p3) ---")
    print(h4_untuned.round(5).to_string(index=False))
    print("\n--- Tuned threshold (p4) ---")
    print(h4_tuned.round(5).to_string(index=False))

    h4_untuned.round(5).to_csv("results_v2/table_h4_margin_untuned.csv", index=False)
    h4_tuned.round(5).to_csv("results_v2/table_h4_margin_tuned.csv", index=False)

    print("\nWrote results_v2/table_h3_vs_maslej_{untuned,tuned}.csv, "
          "table_h4_margin_{untuned,tuned}.csv")
    print(f"\nNote: our test set is 63,662 comments; Maslej's own study used 63,978 "
          f"(316 fewer -- short-comment screen applied before training/prediction).")


if __name__ == "__main__":
    main()
