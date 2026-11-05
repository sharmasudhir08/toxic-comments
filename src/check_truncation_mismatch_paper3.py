"""
Paper-3 Stage 1.3 (E3): word-vs-subword-token truncation mismatch at the 128-unit limit, using
Paper-3's own ACTUAL family-specific preprocessing (not the Mac project's uniform "light
cleaning"): deep cleaning (`clean_text_oct26`) for the recurrent-model word count, and RAW
(uncleaned) text for the transformer tokenizers, matching `cleaning="none"` for that family
(pre_registration.md Sec. 5.3).

Read-only: tokenizes the already-saved test set text. No training, no new predictions.

Usage: python -m src.check_truncation_mismatch_paper3   (run from E:\\Toxic Comments)
"""
import os

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

from src.data import load_official_test
from src.text_cleaning import clean_text_oct26


def main():
    test_df = load_official_test("data")
    # Screen 2 (>=3 words) applied before training/prediction -- match the saved 63,662-row set.
    test_df = test_df[test_df["comment_text"].str.split().str.len() >= 3].reset_index(drop=True)
    raw_texts = test_df["comment_text"].tolist()
    print(f"Test set: {len(raw_texts)} comments (matching the saved 63,662-row prediction set)")

    cleaned_texts = [clean_text_oct26(t) for t in raw_texts]
    word_counts = np.array([len(word_tokenize(t, preserve_line=True)) for t in cleaned_texts])
    print(f"\nWord-token counts (deep-cleaned, recurrent-model tokenizer): "
          f"mean={word_counts.mean():.1f}  median={np.median(word_counts):.0f}  max={word_counts.max()}")
    pct_over_128_words = (word_counts > 128).mean() * 100
    print(f"Comments exceeding 128 words (truncated for recurrent models): {pct_over_128_words:.2f}%")

    rows = [{"tokenizer": "NLTK (recurrent, deep-cleaned)", "checkpoint": "-", "n_sampled": len(raw_texts),
             "mean_word_count": word_counts.mean(), "median_word_count": float(np.median(word_counts)),
             "max_word_count": int(word_counts.max()), "mean_subword_per_word_ratio": np.nan,
             "pct_over_128": pct_over_128_words, "words_covered_by_128_budget": 128.0}]

    from transformers import AutoTokenizer
    for ckpt, label in [("bert-base-uncased", "BERT"), ("distilbert-base-uncased", "DistilBERT"),
                         ("roberta-base", "RoBERTa")]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(raw_texts), size=5000, replace=False)
        # Transformers see the RAW, uncleaned text (cleaning="none"). The word-count denominator
        # for the subword-per-word ratio must come from that SAME raw text (simple whitespace
        # split), not the deep-cleaned word count -- otherwise raw text's punctuation/numbers
        # inflate the subword count against a denominator that already excludes them, badly
        # overstating the ratio (caught: an early version of this script gave an implausible
        # 6.5x ratio for BERT by doing exactly that).
        sample_texts = [raw_texts[i] for i in sample_idx]
        sample_raw_word_counts = np.array([len(t.split()) for t in sample_texts])

        subword_counts = np.array([len(tok.tokenize(t)) for t in sample_texts])
        ratio = subword_counts / np.maximum(sample_raw_word_counts, 1)
        sample_word_counts = sample_raw_word_counts  # used below for the >=128-word filter
        pct_over_128_subword = (subword_counts > 128).mean() * 100

        long_enough = sample_word_counts >= 128
        if long_enough.sum() > 0:
            avg_ratio_long = ratio[long_enough].mean()
            words_covered_by_128_subwords = 128 / avg_ratio_long
        else:
            words_covered_by_128_subwords = float("nan")

        print(f"\n{label} ({ckpt}) tokenizer on RAW text, n={len(sample_texts)} sampled comments:")
        print(f"  mean subword-tokens-per-(deep-cleaned)-word ratio: {ratio.mean():.3f}")
        print(f"  comments exceeding 128 subword tokens (truncated for {label}): {pct_over_128_subword:.2f}%")
        print(f"  for comments >=128 words long: 128 subword tokens covers ~{words_covered_by_128_subwords:.1f} words")

        rows.append({"tokenizer": label, "checkpoint": ckpt, "n_sampled": len(sample_texts),
                     "mean_word_count": sample_word_counts.mean(),
                     "median_word_count": float(np.median(sample_word_counts)),
                     "max_word_count": int(sample_word_counts.max()),
                     "mean_subword_per_word_ratio": ratio.mean(),
                     "pct_over_128": pct_over_128_subword,
                     "words_covered_by_128_budget": words_covered_by_128_subwords})

    df = pd.DataFrame(rows)
    os.makedirs("results_v2", exist_ok=True)
    out_path = "results_v2/table_truncation_mismatch.csv"
    df.round(4).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
