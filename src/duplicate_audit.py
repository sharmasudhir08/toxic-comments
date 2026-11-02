"""
TF-IDF near-duplicate audit between train.csv and test.csv (pre_registration.md Sec. 5.1, Screen 1).

For each test comment, finds the maximum cosine similarity (TF-IDF, word unigrams, 100k vocab) against
every training comment. Comments with cosine >= 0.9 are flagged as near-duplicates. This is a one-time,
read-only audit -- it does not change data/train.csv or data/test.csv, and does not affect model
training or prediction (near-duplicate removal is a reporting-stage filter applied when computing
metrics, not a training-time filter -- see pre_registration.md H1).

    python -m src.duplicate_audit            # writes data/duplicate_screen.npz

Output file data/duplicate_screen.npz has three arrays, aligned by row:
    test_id            test.csv "id" column
    cosine_vs_train     max TF-IDF cosine similarity to any training comment
    is_near_duplicate   cosine_vs_train >= 0.9
"""

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def l2norm(X):
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return X.multiply(1.0 / norms[:, None]).tocsr()


def run(data_dir="data", out_path=None, threshold=0.9, max_features=100_000, batch=2000):
    t0 = time.time()
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), sublinear_tf=True)
    vec.fit(pd.concat([train["comment_text"], test["comment_text"]], ignore_index=True))
    Xtr = l2norm(vec.transform(train["comment_text"]))
    Xte = l2norm(vec.transform(test["comment_text"]))
    Xtr_T = Xtr.T.tocsr()

    best_sim = np.zeros(len(test), dtype=np.float32)
    for start in range(0, len(test), batch):
        end = min(start + batch, len(test))
        sims = Xte[start:end].dot(Xtr_T).tocsr()
        for i in range(sims.shape[0]):
            row = sims.getrow(i)
            if row.nnz:
                best_sim[start + i] = row.data.max()

    mask = best_sim >= threshold
    print(f"near-duplicate test rows (cosine >= {threshold}): {mask.sum()} / {len(test)}  "
          f"({time.time() - t0:.1f}s)")

    out_path = out_path or f"{data_dir}/duplicate_screen.npz"
    np.savez_compressed(out_path, test_id=test["id"].values, cosine_vs_train=best_sim,
                         is_near_duplicate=mask)
    print(f"saved {out_path}")
    return best_sim, mask


if __name__ == "__main__":
    run()
