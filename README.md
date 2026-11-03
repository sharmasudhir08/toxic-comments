# toxic-comments

Code and results for a manuscript comparing six models — LSTM, Bi-LSTM, GRU
(recurrent) and BERT, DistilBERT, RoBERTa (transformer) — on multi-label toxic comment
classification, using the Jigsaw Toxic Comment Classification Challenge dataset. Three random
seeds per model. Licensed under Apache License 2.0.

## Results at a glance

Tuned-threshold, mean ± SD over 3 seeds (full table with all 16 metrics and the default-threshold
rows: `tables/table1_overall_mean_std.csv`):

| Model | F1 (macro, M) | ROC-AUC (macro) | MCC (macro) | Train min/run |
|---|---|---|---|---|
| LSTM | 0.5738 ± 0.0036 | 0.9789 ± 0.0027 | 0.5667 ± 0.0093 | 6.7 |
| Bi-LSTM | 0.5849 ± 0.0088 | 0.9800 ± 0.0018 | 0.5767 ± 0.0055 | 8.0 |
| GRU | 0.5831 ± 0.0075 | 0.9800 ± 0.0014 | 0.5727 ± 0.0081 | 5.7 |
| BERT | 0.6235 ± 0.0053 | 0.9825 ± 0.0019 | 0.6141 ± 0.0013 | 76.2 |
| DistilBERT | 0.6226 ± 0.0063 | 0.9840 ± 0.0007 | 0.6135 ± 0.0024 | 60.2 |
| RoBERTa | 0.6144 ± 0.0015 | 0.9827 ± 0.0009 | 0.6054 ± 0.0006 | 73.0 |

## Verifying the paper's numbers

Every table, figure, and statistic reported in the manuscript traces to a file in this repository.
`TRACEABILITY.md` maps each manuscript claim to its source file, so anyone can check any number
without retraining anything. File integrity can be checked against `SHASUMS`
(`sha256sum -c SHASUMS`).

**Example — verifying the H8 calibration claim** (minority labels are better-calibrated than
frequent labels, for every model): run `python -m src.calibration_analysis` from the repository
root, against the saved per-seed outputs. Its printed Holm-corrected p-values should match
`tables/table_h8_calibration_equal_width_console_log.txt` exactly, and its summary numbers match
`tables/table_h8_calibration_equal_width.csv`. The same pattern (run the named script in `src/`,
compare its output against the matching file in `tables/`) applies to every other claim listed in
`TRACEABILITY.md`.

## Layout

- `src/` — data loading/preprocessing, model training (recurrent and transformer pipelines), and
  every statistical analysis script used in the paper (significance testing, calibration,
  seed-stability, length-robustness).
- `tables/` — every result CSV and the raw console-log text for analyses that otherwise only
  print to screen, kept here so each number has a saved, checkable source.
- `figures/` — the architecture diagrams and result figures used in the manuscript, at the
  resolution submitted.
- `TRACEABILITY.md` — the manuscript-claim-to-file map described above.

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`.
2. Download the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle and place
   `train.csv`, `test.csv`, `test_labels.csv` in `data/` (not included in this repository; see
   Kaggle's terms).
3. Train: recurrent models via `python -m src.rnn_proposed`, transformer models via
   `python -m src.run_experiments` (3 seeds each, both families).
4. Run the analysis scripts in `src/` (e.g. `src.calibration_analysis`, `src.length_robustness`,
   `src.h2_full_methods`) against the saved per-seed outputs to regenerate every table in
   `tables/`.

## Not included

The dataset itself is not redistributed (download separately from Kaggle). A Python virtual
environment is not included; regenerate it from `requirements.txt`.
