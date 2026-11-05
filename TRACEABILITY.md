# Traceability: paper item → result file → script

Every number in `results/` is produced by exactly one script in `src/` (or `generate_results.py`)
reading already-saved model outputs. No number here is hand-typed or hand-edited. See
`results/MAPPING.csv` for the same information as a machine-readable CSV.

| Paper item | Result file | Script |
|---|---|---|
| Table 1 (label distribution) | `results/Table01_label_distribution.csv` | computed directly from the raw Kaggle data files, no model output needed |
| Table 2 (family-specific settings) | `results/Table02_family_design_choices.csv` | transcribed from the manuscript (settings, not computed) |
| Table 3 (overall perf., Acc/F1/AUC) | `results/Table03_overall_performance_acc_f1_auc.csv` | `generate_results.py --results_dir results_proposed` |
| Table 4 (overall perf., MCC/ECE/Brier/time) | `results/Table04_overall_performance_mcc_ece_brier_time.csv` | `generate_results.py --results_dir results_proposed` (same source file as Table 3) |
| Table 5 (per-label F1) | `results/Table05_per_label_f1.csv` | `generate_results.py --results_dir results_proposed` |
| Table 6 (H1 primary, nested ANOVA) | `results/Table06_h1_nested_anova_primary.csv` | `python -m src.h1_nested_anova_paper3` |
| Table 6 (H1 secondary, Welch/permutation) | `results/Table06_h1_welch_permutation_secondary.csv` | `python -m src.family_comparison` |
| Table 7 (H3a/H3b vs. benchmark, tuned) | `results/Table07_h3_benchmark_tuned.csv` | `python -m src.h3_h4_paper3` |
| Table 8 (H7 per-bin nested ANOVA) | `results/Table08_h7_per_bin_nested_anova.csv` | `python -m src.h7_length_control_nested_anova_paper3` |
| Table 9 (H8 Brier Skill Score) | `results/Table09_h8_brier_skill_score.csv` | `python -m src.e1_prevalence_normalized_calibration_paper3` |
| Table 10 (outcome synthesis) | — (collects Tables 3-9, A3-A14; no separate source) | — |
| Table A1 (hyperparameters) | `results/TableA01_hyperparameters.csv` | transcribed from the manuscript (settings, not computed) |
| Table A2 (hardware/software) | `results/TableA02_environment.csv` | transcribed from the manuscript (settings, not computed) |
| Table A3 (full 48-row benchmark) | `results/TableA03_benchmark_48row_full.csv` | `python -m src.h3_h4_paper3` (per-row detail), pooled Holm recomputed across the 48-test family |
| Table A4 (Tukey HSD, ROC-AUC) | `results/TableA04_tukey_hsd_roc_auc.csv` | `python -m src.h2_full_methods` |
| Table A5 (Tukey HSD, F1(M)) | `results/TableA05_tukey_hsd_f1m.csv` | `python -m src.h2_full_methods` |
| Table A6 (DeLong, median p) | `results/TableA06_delong_median.csv` | `python -m src.h2_full_methods` |
| Table A7 (DeLong, Fisher-combined) | `results/TableA07_delong_fisher_combined.csv` | `python -m src.h2_delong_fisher_corrected` |
| Table A8 (paired bootstrap) | `results/TableA08_paired_bootstrap.csv` | `python -m src.h2_full_methods` |
| Table A9 (tuned thresholds) | `results/TableA09_tuned_thresholds.csv` | `generate_results.py --results_dir results_proposed` |
| Table A10 (per-label margin) | `results/TableA10_per_label_margin_vs_benchmark.csv` | `generate_results.py --results_dir results_proposed` |
| Table A11 (H8 adaptive-binning retest) | `results/TableA11_h8_adaptive_binning_retest.csv` | `python -m src.calibration_adaptive_binning` |
| Table A12 (H5 single-vs-multi-seed) | `results/TableA12_h5_single_vs_multi_seed.csv` | `python -m src.single_vs_multi_seed` |
| Table A12 (H5 paired permutation) | `results/TableA12_h5_paired_permutation.csv` | not currently a standalone script in this repository — see note below |
| Table A13 (H9 CV ratios) | `results/TableA13_h9_cv_ratios.csv` | base CV values from `python -m src.seed_stability`; the ratio columns are not currently a standalone script in this repository — see note below |
| Table A14 (H10 truncation) | `results/TableA14_h10_truncation.csv` | `python -m src.check_truncation_mismatch_paper3` |
| Fig. 1 (workflow) | `figures/Fig1.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 2 (recurrent architecture) | `figures/Fig2.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 3 (transformer architecture) | `figures/Fig3.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 4 (overall performance) | `figures/Fig4.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 5 (calibration + BSS) | `figures/Fig5.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 6 (length robustness) | `figures/Fig6.png` | extracted directly from the final manuscript; no source script in this repository |

## Honest gaps

Three result files currently have saved data but no standalone, re-runnable script in this
repository:

- **`Table 8 (H7 per-bin nested ANOVA)`**, specifically the `p_holm_10tests` and `sig_holm_primary`
  columns — `src/h7_length_control_nested_anova_paper3.py` reproduces all other columns (bin,
  n_comments, metric, recurrent_mean, transformer_mean, diff, F_family_1_4, p_family_1_4) exactly;
  verified by re-running it and diffing the output against `results/Table08_h7_per_bin_nested_anova.csv`
  (2026-11-05). The two Holm-correction columns are present in the saved CSV but are not produced
  by the current version of that script, and no other script in this repository computes them.
- **`Table A12 (H5 paired permutation)`** — the exact-permutation robustness check on seed-averaged
  correctness. The underlying McNemar+Fisher test (`src/single_vs_multi_seed.py`) is included and
  reproducible; the permutation check itself was computed with ad-hoc code during analysis and has
  not yet been packaged as a standalone script.
- **`Table A13 (H9 CV ratios)`**, specifically the `ece_over_f1_ratio` and `brier_over_auc_ratio`
  columns — `src/seed_stability.py` computes and prints the underlying CV(F1), CV(AUC), CV(ECE)
  and CV(Brier) values (reproducible), but the ratio columns themselves were computed with ad-hoc
  code, not a saved script.

Both are flagged here rather than silently omitted or backed by a script written after the fact
that might not exactly match the saved numbers. The raw CV values in `src/seed_stability.py`'s
output do match the saved CSV (verify by running it and comparing the printed percentages).

## No reproduction claim without the saved model outputs

None of the scripts above can be run without the saved per-model, per-seed test-set probabilities
(and, for H1, the near-duplicate screen mask). These are not included in this repository (see
`README.md`); request them from the authors to actually execute the reproduction steps above. File
integrity for everything in `results/` can still be checked against `SHASUMS` without them.
