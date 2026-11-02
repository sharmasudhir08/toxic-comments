"""
Label distribution table (train + official scored test set), computed directly from the
raw data files -- not copied from any paper or prior document.

Produces the exact table required in both journal papers' Methods sections
(pre_registration.md Sec 10, Preregistration-2.md Sec 5): per-label counts and
percentages, "No label" and ">=2 labels" rows, for train.csv and the scored portion
of test.csv + test_labels.csv.

Usage: python -m src.label_distribution
"""
import pandas as pd

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def summarize(df, name):
    n = len(df)
    rows = []
    for c in LABELS:
        cnt = int(df[c].sum())
        rows.append((c, cnt, 100 * cnt / n))
    no_label = int((df[LABELS].sum(axis=1) == 0).sum())
    multi_label = int((df[LABELS].sum(axis=1) >= 2).sum())
    rows.append(("No label", no_label, 100 * no_label / n))
    rows.append((">=2 labels", multi_label, 100 * multi_label / n))
    rows.append(("Total", n, 100.0))
    print(f"--- {name}: n={n} ---")
    for label, cnt, pct in rows:
        print(f"  {label:15s} {cnt:7d}  {pct:.2f}%")
    return rows


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    labels = pd.read_csv("data/test_labels.csv")

    test_merged = test.merge(labels, on="id", how="inner")
    test_scored = test_merged[(test_merged[LABELS] != -1).all(axis=1)].reset_index(drop=True)

    train_rows = summarize(train, "TRAIN")
    test_rows = summarize(test_scored, "TEST (scored)")
    return train_rows, test_rows


if __name__ == "__main__":
    main()
