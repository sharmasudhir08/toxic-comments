# toxic-comments

Reviewer-verification package for a manuscript comparing six models — LSTM, Bi-LSTM, GRU
(recurrent) and BERT, DistilBERT, RoBERTa (transformer) — on multi-label toxic comment
classification, using the official Jigsaw Toxic Comment Classification Challenge dataset. Three
random seeds per model. Licensed under Apache License 2.0.

## Purpose

This repository lets a reviewer check every number reported in the paper against its source: the
statistical test script that computed it, run on the authors' saved model outputs. It is **not**
a training repository — no dataset, no model checkpoints and no saved predictions are included
(see "Saved model outputs" below). Nothing here needs to be retrained to verify a number; every
script in `src/` reads already-computed, saved artifacts.

## Test sets

- **63,662 comments** — the official Jigsaw scored test set (63,978 comments) after removing 316
  comments of two words or fewer (the short-comment screen, applied before training). This is the
  test set used for every reported result except H1's primary test.
- **61,701 comments** — the above set, additionally screened for near-duplicates of the training
  set (TF-IDF cosine similarity ≥ 0.9; 1,961 comments removed). Used **only** for H1's primary
  nested-ANOVA test (pre-registered as primary for the family-level comparison specifically); H1's
  secondary test and H2–H10 use the full 63,662-comment set.

## Hypotheses and which script tests each

| Hyp. | What it tests | Script(s) in `src/` |
|---|---|---|
| H1 (primary) | Transformers vs. recurrent, family level, nested ANOVA, screened set (61,701) | `h1_nested_anova_paper3.py` |
| H1 (secondary) | Same, Welch's t-test + exact permutation, full set (63,662) | `family_comparison.py` |
| H2 | Pairwise cross-family, Tukey HSD + Holm (primary), DeLong's test, paired bootstrap | `h2_full_methods.py`, `h2_delong_fisher_corrected.py` (Fisher-combined robustness check) |
| H3a/H3b | vs. published benchmark (Maslej-Kresnakova et al. 2020), tuned and untuned comparators, pooled 48-test Holm correction | `h3_h4_paper3.py`, `generate_results.py` |
| H4 | Minority- vs. frequent-label margin over the benchmark | `h3_h4_paper3.py`, `generate_results.py` |
| H5 | Single-seed vs. multi-seed significance (McNemar + Fisher), paired permutation robustness check | `single_vs_multi_seed.py` |
| H6 | Family-level calibration (ECE) + training-cost ratio | `h1_nested_anova_paper3.py` (ECE row), `generate_results.py` (training time) |
| H7 | Length-bin robustness, per-bin nested ANOVA | `h7_length_control_nested_anova_paper3.py` (Holm correction over the 10 tests: see "Honest gaps" in `TRACEABILITY.md`) |
| H8 (post-hoc) | Minority-vs-frequent raw ECE (paired t-test) and Brier Skill Score | `h8_raw_ece_paired_ttest_save.py`, `e1_prevalence_normalized_calibration_paper3.py`, `calibration_adaptive_binning.py` (adaptive-binning retest) |
| H9 (post-hoc) | Seed-sensitivity (coefficient of variation) of calibration vs. discrimination metrics | `seed_stability.py` |
| H10 (post-hoc) | Word/subword truncation mismatch at the 128-unit budget | `check_truncation_mismatch_paper3.py` |

Every hypothesis's pre-registered or post-hoc wording is given in full in the manuscript, Sec. 1.4
and 1.4b.

## Verifying a number

1. Find the number in `results/MAPPING.csv` (paper item → result file → source CSV name).
2. Run the matching script from this repository's root (see table above), against the saved
   outputs you have been given access to (see below).
3. Compare its printed or saved output to `results/<FileName>.csv`.

## Saved model outputs

Saved model outputs (per-model, per-seed test-set probabilities, and the near-duplicate screen
mask used for H1) are **not included in this repository** to keep it small and because they are
derived from licensed Kaggle data. They are **available from the authors on reasonable request**.
With them in place (see each script's own docstring for the expected path), every script in
`src/` reproduces its result file exactly, with no training or new inference.

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`.
2. Download the official Jigsaw Toxic Comment Classification Challenge dataset from Kaggle
   (`train.csv`, `test.csv`, `test_labels.csv`) — needed only if you want to re-derive the
   near-duplicate/short-comment screens or the length/truncation statistics from raw text; most
   statistical tests need only the saved model outputs above.
3. Obtain the saved model outputs from the authors (see above) and place them at the path each
   script expects.
4. Run `python generate_results.py --results_dir results_proposed` for the overall-performance,
   per-label and benchmark-margin tables, or any individual script in `src/` listed above for a
   specific hypothesis.

## Layout

- `src/` — the statistical analysis and significance-test scripts behind every reported number,
  plus the small shared utilities they import (`config.py`, `data.py`, `metrics.py`,
  `preprocessing.py`, `model_registry.py`, `maslej_baselines.py`, `paper_baselines.py`,
  `text_cleaning.py`). No training code.
- `generate_results.py` — produces the overall-performance, per-label and (untuned) benchmark
  tables from saved per-run metrics.
- `results/` — every result CSV behind the paper's tables, named by final table number, plus
  `MAPPING.csv` (paper item → file → source).
- `figures/` — the six figures as they appear in the final manuscript, extracted directly from it.
- `TRACEABILITY.md` — table/figure number → result file → script.

## Not included

The dataset, model checkpoints, saved predictions, and the manuscript file itself (`.docx`/`.docm`)
are not included. Figure-generation and manuscript-preparation scripts are intentionally excluded
from this reviewer-check package (statistical-analysis scripts only).
