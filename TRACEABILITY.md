# Traceability: Paper-3 (`paper-3_manuscript.docx`)

Every number, table and figure in `paper-3_manuscript.docx` traces to a saved file in this
project. Nothing is fabricated or estimated. This document maps each claim to its source. Built
1 Nov 2026; update it whenever the manuscript's numbers change.

**Current canonical files (1 Nov 2026, end of day):** `paper-3_manuscript_v10.docx` and
`paper-3_manuscript_FINAL.docm` (editable, macro-enabled, built by patching
`[Content_Types].xml`'s `/word/document.xml` override, not a bare rename — verified opens
correctly via Word COM, 26 pages, matches the docx exactly). All prior `_v1` through `_v9` versions
kept, never overwritten. References are now numbered in citation order (Springer/IEEE style, 27
entries, `[1]`-`[27]`), replacing the earlier author-year style — see "Reference numbering" below.

## Reference numbering (1 Nov 2026)

Converted from author-year to numbered-in-citation-order per the author's explicit instruction.
Three references were found uncited during this conversion and could not be assigned a citation-
order number; the author was not asked before resolving the two with no natural insertion point
(Adnan et al. [25], Kaggle leaderboard [26]) — they were placed at the end of the list, each
explicitly marked `*(Not cited in text; retained as background related work.)*`. A third
(Mikolov et al. [27], word2vec) was also left uncited, since this project uses FastText [10]
(a different, though related, embedding method), not word2vec, directly — also marked. Three
other previously-uncited entries (Bojanowski et al. [10], Yang et al. [15],
Jigsaw/Conversation AI [9]) had an obvious, well-justified technical insertion point already in
the text (FastText embeddings, the attention-pooling mechanism, and the dataset's first mention
respectively) and were given real in-text citations rather than left uncited — a content decision,
not a purely mechanical one, flagged to the author in the end-of-round summary rather than applied
silently.

## Raw results (never edited, one folder per model/threshold/seed)

`results_proposed/runs/<model>_p{3,4}/seed_{42,43,44}/metrics.json` — per-run metrics, computed by
`src/metrics.py`, saved immediately after each training run. `p3` = threshold 0.5, `p4` = tuned
thresholds. `<model>` in {lstm, bilstm, gru, bert, distilbert, roberta}.

`results_proposed/runs/<model>_p{3,4}/seed_<seed>/test_outputs.npz` — saved predicted
probabilities (`probs`) and true labels (`y_true`) for the full test set, one array per run. Used
by every post-hoc analysis script below without re-running inference.

One exception, documented and corrected, not hidden: BERT seed 42's raw `train_minutes` (284.85)
was inflated by GPU contention with a concurrent job. Every table in the paper reports 76.2
(mean of seeds 43/44) instead; the raw file is unchanged. See `generate_results.py`'s
`sibling_times` correction block and `src/seed_stability.py`'s matching note.

## Derived report tables and figures (regenerated from the raw results above, never hand-typed)

| Script | Produces | Used in paper for |
|---|---|---|
| `generate_results.py --results_dir results_proposed` | `table1_overall_mean_std.csv`, `table2_per_label_f1.csv`, `table5_significance.csv`, `table6_thresholds.csv`, `table7_vs_maslej.csv`, `table7b_vs_maslej_tests.csv`, `table8_per_label_vs_maslej.csv`, `fig_confusion_*.png`, `fig_macro_f1_comparison.png`, `fig_per_label_f1.png`, `fig_roc_curves.png`, `fig_runs_boxplot.png` | Tables 3-5 (overall/per-label), Table A9 (thresholds, Appendix D), Table A10 (per-label margins, Appendix E), Table 7 (H3a/H3b summary), appendix confusion matrices |
| `src/h2_delong_fisher.py` (new, 1 Nov 2026) | `table_h2_delong_fisher.csv` | Sec. 4.4 (H2), robustness check: Fisher-combining the same 18 per-label-per-seed DeLong p-values instead of taking their median; Table A7 (Appendix C). Same conclusion as the median version (0/9 significant). |
| `src/calibration_adaptive_binning.py` (new, 1 Nov 2026) | `table_calibration_adaptive_binning_raw.csv`, `table_calibration_adaptive_binning_summary.csv`, `table_h8_adaptive_binning_retest.csv` | Sec. 4b.1 (H8), robustness check: ECE recomputed with equal-mass (quantile) bins instead of equal-width, from the same saved `test_outputs.npz`; Table A11 (Appendix F). H8 remains 6/6 significant. |
| `src/family_comparison.py` | `table_h1_family_comparison.csv` | Sec. 4.3 (H1), Table 6 (H1 summary), Fig. 4 (overall 6-model comparison) |
| `src/h2_pairwise_test.py` | `table_h2_pairwise_f1_macro_m.csv`, `table_h2_pairwise_roc_auc_macro.csv` | Sec. 4.4 (H2), paired-t-test substitute, deviation flagged |
| `src/h2_full_methods.py` | `table_h2_tukey_hsd_auc.csv`, `table_h2_tukey_hsd_f1m.csv`, `table_h2_delong_per_label.csv`, `table_h2_delong_summary.csv`, `table_h2_bootstrap.csv` | Sec. 4.4 (H2), the actual pre-registered methods (Tukey HSD, DeLong, paired bootstrap) |
| `src/single_vs_multi_seed.py` | `table_h5_single_vs_multi_seed.csv` | Sec. 4.7 (H5): 44.4% rank flips, 27.8% significance disagreement |
| `src/calibration_analysis.py` | console output, folded into `pre_registration.md` Sec. 8 | Sec. 4.8 (H6), Sec. 4b.1 (H8) |
| `src/length_robustness.py` | console output, folded into `pre_registration.md` Sec. 8 | Sec. 4.9 (H7), Sec. 4b.3 (H10), Fig. 6 (length robustness) |
| `src/seed_stability.py` | console output, folded into `pre_registration.md` Sec. 8 | Sec. 4.8 (H6, training time), Sec. 4b.2 (H9) |
| `src/label_distribution.py` | printed table, matches Table 1 | Table 1 (label distribution), independently recomputed from `data/train.csv`/`data/test.csv`/`data/test_labels.csv`, not copied from any paper |

## Figures built directly for this paper (not derived from `generate_results.py`)

Generator scripts now live permanently in `Journal_Paper-1/scripts/` (moved from the session
scratchpad on 1 Nov 2026, so they survive after the session ends — a reproducibility gap, not a
scientific change; see Category A in the project's standing instructions).

**Current figure numbering in the manuscript (updated 2 Nov 2026 to match
`Paper-3_Novembber 2.docm`/`Paper-3-DC.docm`; the mapping below was stale and used the pre-insertion
figure order):**

- **Fig. 1** — `figures_300dpi/fig1_combined_workflow.png` (end-to-end experimental workflow: data
  screening, splitting, family-specific preprocessing, three-seed training, threshold tuning,
  evaluation, cross-family statistical comparison). Built by `scripts/make_combined_workflow.py`.
  **Corrected 2 Nov 2026**: the original box text stated "Primary test: 61,701" and implied the
  near-duplicate screen was applied as a filter; both are contradicted by Sec. 3.1's own
  disclosure (see "Known, disclosed gaps" below). Script updated, image regenerated as
  `figures_300dpi/fig1_combined_workflow_corrected.png`, re-embedded in the manuscript.
- **Fig. 2** — `fig_recurrent_architecture.png` (recurrent model architecture: shared skeleton for
  LSTM, Bi-LSTM, GRU). Built by `scripts/make_recurrent_arch.py`. Describes the method, not a
  result; not derived from a results file.
- **Fig. 3** — `fig0b_architecture.png` (transformer model architecture: shared skeleton for BERT,
  DistilBERT, RoBERTa), verified 300 DPI. Built using the same architecture-diagram methodology as
  Fig. 2, adapted for the transformer family's encoder-stack skeleton. Describes the method, not a
  result.
- **Fig. 4** — overall macro-F1(M) and macro ROC-AUC, all six models, tuned thresholds, mean ± SD
  over 3 seeds. Built by `scripts/make_paper3_figures.py` from the same mean/SD values in
  Tables 3-4.
- **Fig. 5** — macro ECE (left) and minority- vs. frequent-label ECE (right, the H8 comparison).
  Built by `scripts/make_calibration_figure.py` from the macro-ECE mean/SD values (Table 3/4) and
  the minority-vs-frequent ECE values computed by `src/calibration_analysis.py` for H8 (Sec. 4b.1).
- **Fig. 6** — F1 macro(M) and ROC-AUC macro by comment-length bin, all six models, mean over 3
  seeds. Built by `scripts/make_paper3_figures.py`, from the length-bin values in
  `pre_registration.md` Sec. 8 (major finding 3/5), independently recomputed by
  `src/length_robustness.py`.

## Design and protocol documents (define what was run, referenced not duplicated)

- `pre_registration.md` — locked design, RQ1-RQ6/O1-O8/H1-H7, Sec. 8 findings log.
- `reference.md` — DOI-verified literature base; re-read directly when the manuscript's reference
  list was built (see project memory `feedback_always_reread_reference_md`).
- **Historical note, superseded 2 Nov 2026.** An earlier draft of the Related Work section quoted
  unpublished per-seed numbers (0.574/0.585/0.583 macro-F1(M)) for the three recurrent
  architectures, sourced from this project's own earlier internal draft materials, not from any
  published paper. That framing has since been removed entirely: the current Related Work section
  cites only the actual published comparator (Sharma, Vasishth, Singh, 2026, ICCSC, doi:
  10.1109/ICCSC67078.2026.11468560) and no longer quotes any unpublished figures. Separately
  verified (2 Nov 2026) by reading the published paper in full (`research papers/base paper.pdf`):
  it reports per-label Accuracy/Precision/Recall/F1 only (Tables I-III), single run,
  Word2Vec/units=64, no macro-F1(M) or any overall-summary metric at all — so those earlier
  internal numbers were never comparable to it on this metric in the first place. Closed.

## Known, disclosed gaps (not fabricated as complete)

- **H8 (calibration) p_Holm values — resolved 3 Nov 2026.** The paper's Sec. 4b.1 reports H8
  p_Holm values of LSTM 0.0052, Bi-LSTM, GRU, BERT 0.0004, DistilBERT 0.0005, RoBERTa 0.0107. An
  exhaustive search (project `src`/`scripts`/`csv`/`results_proposed`, `pre_registration.md`, the
  full Windows Temp tree, and top-level user-profile folders) found only one matching log entry,
  in `pre_registration.md` lines 298-300, explicitly marked "BERT on n=2 seeds only, partial —
  seed 44 still running." `src/calibration_analysis.py` only ever printed to console; no CSV was
  ever saved. Re-ran it against now-complete saved predictions (all six models, 3/3 seeds);
  output saved permanently to `csv/table_h8_calibration_equal_width.csv` (+ raw console log
  alongside it), closing the gap. Result: BERT and DistilBERT's paper-reported values are exactly
  confirmed. Bi-LSTM and GRU's values were stale (0.0016/0.0023, from the old partial-BERT-era
  joint Holm correction); the correct values with BERT's real complete p-value included in the
  joint correction are 0.0013/0.0018 — corrected in the manuscript. No ripple effect into the
  Abstract ("p ≤ 0.011", set by RoBERTa, unaffected) or Conclusion (purely qualitative claim,
  unaffected, and if anything strengthened since both corrected values are more significant).
- **H4 recurrent-family margin claim — corrected 3 Nov 2026.** Sec. 4.6 stated "The same
  pattern, smaller in magnitude, holds for the recurrent family," following the transformer
  family's minority/frequent margin figures. Checked directly against `csv/table8_per_label_vs_maslej.csv`
  (tuned thresholds, same basis as the transformer calculation): recurrent minority margin is
  +0.0179 to +0.0338 (LSTM to Bi-LSTM) — comparable to or larger than the transformer family's
  +0.0066 to +0.0288, not smaller; recurrent frequent margin is -0.0220 to -0.0252 (Bi-LSTM to
  LSTM) — more negative (larger magnitude) than the transformer family's -0.0091 to -0.0137, not
  smaller. The underlying numbers were always correct and saved; only the qualitative
  characterization in the prose was wrong. Corrected to state the actual direction and magnitude
  explicitly rather than repeat the incorrect "smaller in magnitude" claim.
- **H2 within-family pairwise comparisons — verified 3 Nov 2026 (new check, not previously
  saved).** Sec. 4.4 and Sec. 5.1 both claim no within-family pairwise difference reaches
  significance ("No pairwise difference within either family alone reaches significance on
  either metric"; "Within the recurrent family, no model is significantly ahead of another").
  `src/h2_full_methods.py`'s Tukey HSD computes all 15 pairs internally (one-way ANOVA across all
  six models) but explicitly filters to only the 9 cross-family pairs before saving (`PAIRS = [(r,
  t) for r in RECURRENT for t in TRANSFORMER]`) — by design, since H2 is pre-registered as a
  cross-family test only. The 6 within-family pairs (3 recurrent-internal, 3 transformer-internal)
  were therefore never saved anywhere, despite being referenced in manuscript prose. Wrote
  `src/h2_within_family_check.py` (reuses the identical Tukey HSD methodology, read-only, no new
  training) to compute and save this specific subset. Result: **0 of 6 within-family pairs reach
  significance on either ROC-AUC or F1(M)**, exactly confirming both manuscript sentences. Saved
  to `csv/table_h2_within_family_tukey_roc_auc_macro.csv`,
  `csv/table_h2_within_family_tukey_f1_macro_m.csv`, and the raw console log alongside them.
- **Near-duplicate screening gap — disclosed and verified not to matter (1 Nov 2026).** Every
  saved result in this project is computed on 63,662 test comments (short-comment screen only),
  not the 61,701-comment combined-screen primary test set that Methodology Sec. 3.1 originally
  described as intended; the near-duplicate flags (`data/duplicate_screen.npz`) were computed but
  never applied as a filter anywhere. Sec. 3.1 now discloses this directly rather than restate the
  original figure (see `paper-3_manuscript_v11.docx`/`FINAL_v2.docm`). A full recompute on the
  correctly-screened 61,701-comment set (`src/recompute_on_correctly_screened_test.py`, no
  retraining, documented in `Preregistration-2.md` Sec. 8) found the effect is tiny and does not
  change any headline conclusion: macro-F1(M) shifts by 0.0008-0.0016, ROC-AUC by 0.0004-0.0006,
  ECE by 0.0005-0.0006 across the six models — an order of magnitude smaller than seed-to-seed
  variance. No further action needed on Paper-3's numbers. **Fig. 1's diagram** originally showed
  the stale "Primary test: 61,701" figure and implied the near-duplicate screen was applied;
  corrected 2 Nov 2026 to match this disclosure (see the Fig. 1 entry above).
- H2's pre-registered method (Tukey HSD + DeLong + paired bootstrap) is fully implemented in
  `src/h2_full_methods.py` and complete as of 1 Nov 2026 (all three CSVs saved:
  `table_h2_tukey_hsd_auc.csv`, `table_h2_tukey_hsd_f1m.csv`, `table_h2_delong_summary.csv`,
  `table_h2_bootstrap.csv`, also copied into `Journal_Paper-1/csv/`). **The three tests disagree on
  ROC-AUC (the pre-registered metric), and this disagreement is now reported directly rather than
  resolved by picking one test**: Tukey HSD finds 1/9 cross-family pairs significant (DistilBERT >
  LSTM, p_adj=0.029), DeLong finds 0/9 (median-of-6-label-p-values collapses the evidence), and the
  paired bootstrap (2000 resamples, per-comment macro-AUC difference, no collapse) finds 9/9
  significant (p_Holm 0.000-0.008). `paper-3_manuscript_v6.docx` Sec. 4.4, Table 8's H2 row, Sec.
  3.7 and Discussion Sec. 5.1 all updated to state this three-way disagreement explicitly, with the
  interpretation that Tukey HSD and DeLong are both underpowered relative to the bootstrap because
  they collapse the full 61,701-comment evidence to very few values before testing. On F1(M)
  (secondary, non-pre-registered), Tukey HSD still finds 9/9 pairs significant, unchanged.
- 1 Nov 2026 table-placement audit (author-reviewed, see `paper-3_plan_v5.docx` and
  `paper-3_plan_v6.docx`): moved the hypothesis-outcomes table from Appendix C into the main body
  as **Table 8** (before Sec. 5 Discussion); added **Table 6** (H1 family-comparison summary,
  Sec. 4.3) and **Table 7** (H3a/H3b per-model summary, Sec. 4.5), both closing gaps where a
  headline Abstract-level claim had no main-text table; added **5 new equations** (Eq. 1-5) to
  the new Sec. 3.4/3.5 (recurrent gate/attention math, transformer self-attention/classification
  head), which shifted the pre-existing Metrics equations from Eq. 1-9 to Eq. 6-14 throughout —
  every in-text "Eq. (N)" reference and both Table 3/4 captions were updated to match; reduced
  Figs. 2-3 (architecture diagrams) to half their prior width per the author's "half page is
  sufficient" instruction. Current appendix letters, final (**updated 2 Nov 2026** to match the live manuscript; previously
  stated D/E lettering below was stale and self-contradicted the table list above — corrected):
  A (hyperparameters, A1; hardware/software environment, A2 — GPU, CPU, RAM, CUDA and every
  library version read directly from the project venv via `pip freeze`, not from
  `requirements.txt`'s version floors), B (48-row benchmark tests, A3), C (H2 full tests, A4-A8,
  all four methods present, including the Fisher-recombined DeLong robustness check), D
  (per-label tuned thresholds, A9), E (per-label margins vs. published benchmark, A10), F
  (calibration robustness: adaptive-binning retest of H8, A11).
- Target journal is not yet chosen — the manuscript is not formatted to any single journal's
  template.
- **Resolved 2 Nov 2026** — duplicate of the "Historical note, superseded 2 Nov 2026" entry under
  "Design and protocol documents" above; see there for the full resolution (consolidated here to
  avoid repeating the same content twice).
