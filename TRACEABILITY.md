# Traceability

Every number, table, and figure in the manuscript traces to a file in this repository. Nothing is
hand-typed or estimated. This document maps each claim to its source.

## Raw results

`results_proposed/runs/<model>_p{3,4}/seed_{42,43,44}/metrics.json` — per-run metrics (F1, ROC-AUC,
MCC, ECE, Brier, per-label breakdowns), saved immediately after each training run completes.
`p3` = default (0.5) threshold, `p4` = validation-tuned thresholds. `<model>` is one of
{lstm, bilstm, gru, bert, distilbert, roberta}. Not included in this repository (training outputs
only; regenerate by training, see README.md's "Reproduction" section); the derived CSVs below are.

`results_proposed/runs/<model>_p{3,4}/seed_<seed>/test_outputs.npz` — saved predicted probabilities
and true labels for the full test set, one array per run. Used by every analysis script below
without re-running inference.

One correction, applied consistently: BERT seed 42's raw training time was inflated by GPU
contention with a concurrent job on the development machine. Every reported training time for
BERT uses the mean of seeds 43 and 44 (76.2 min) instead; no quality metric (F1, AUC, etc.) for
seed 42 is affected.

## Derived tables and figures

| Script | Produces | What it covers |
|---|---|---|
| `src/family_comparison.py` | `table_h1_family_comparison.csv` | Family-level (recurrent vs. transformer) comparison |
| `src/h2_full_methods.py` | `table_h2_tukey_hsd_*.csv`, `table_h2_delong_summary.csv`, `table_h2_bootstrap.csv` | Cross-family pairwise significance: Tukey HSD, DeLong's test, paired bootstrap |
| `src/h2_delong_fisher.py` | `table_h2_delong_fisher.csv` | DeLong's test, Fisher-combined across labels (robustness check on the method above) |
| `src/h2_within_family_check.py` | `table_h2_within_family_tukey_*.csv` | Within-family pairwise significance (3 recurrent-internal + 3 transformer-internal pairs) |
| `src/h6_within_recurrent_ece_check.py` | console log only | Within-recurrent-family calibration (ECE) pairwise test |
| `src/calibration_analysis.py` | `table_h8_calibration_equal_width.csv` | Macro/per-label ECE and Brier score, minority- vs. frequent-label calibration |
| `src/calibration_adaptive_binning.py` | `table_calibration_adaptive_binning_*.csv`, `table_h8_adaptive_binning_retest.csv` | Calibration robustness check: equal-mass binning instead of equal-width |
| `src/seed_stability.py` | console log only | Seed-to-seed variability (coefficient of variation) of discrimination vs. calibration metrics |
| `src/single_vs_multi_seed.py` | `table_h5_single_vs_multi_seed.csv` | Whether single-seed and multi-seed pairwise conclusions agree |
| `src/length_robustness.py` | console log only | Performance by comment-length bin |
| `src/label_distribution.py` | printed table | Label distribution, computed independently from the raw data files |
| `src/duplicate_audit.py` | — | Near-duplicate audit between train and test splits |
| `src/recompute_on_correctly_screened_test.py` | — | Headline metrics recomputed on an alternative, more strictly screened test set (see note below) |
| `src/maslej_baselines.py`, `src/paper_baselines.py` | — | Published comparator results, used for benchmark comparison tables |

CSVs and console logs for every script above are in `tables/`. Figures (architecture diagrams and
result plots) are in `figures/`.

## Notes for correctly interpreting these results

- **Test-set definition.** All results in this repository use a 63,662-comment test set (a
  short-comment screen only). A stricter, additionally near-duplicate-screened 61,701-comment set
  was also evaluated (`src/recompute_on_correctly_screened_test.py`, no retraining): the difference
  is small and does not change any headline conclusion (macro-F1(M) shifts by 0.0008-0.0016,
  ROC-AUC by 0.0004-0.0006, ECE by 0.0005-0.0006 across all six models — an order of magnitude
  smaller than seed-to-seed variance).
- **H2 (cross-family significance on ROC-AUC) — three methods disagree, reported directly rather
  than resolved by picking one.** Tukey HSD finds 1 of 9 cross-family pairs significant; DeLong's
  test (median p-value across labels) finds 0 of 9; a paired bootstrap over individual test
  comments (no collapsing across labels) finds 9 of 9 significant. The manuscript's interpretation
  is that the first two methods are underpowered relative to the bootstrap, since they each
  collapse the full test set's evidence down to very few values before testing. On the secondary
  metric F1(M), all three methods agree (9 of 9 significant).
- **Calibration binning.** ECE is reported with equal-width bins by default. An equal-mass
  (quantile) binning alternative was also tested as a robustness check and gives materially the
  same result (`src/calibration_adaptive_binning.py`).

## Not included

Trained model checkpoints are not included. Raw per-seed training outputs
(`results_proposed/runs/`) are not included; this repository ships only the derived, aggregated
CSVs and console logs built from them, under `tables/`.
