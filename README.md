# toxic-comments

Code and results for a manuscript (under peer review) comparing six models — LSTM, Bi-LSTM, GRU
(recurrent) and BERT, DistilBERT, RoBERTa (transformer) — on multi-label toxic comment
classification, using the Jigsaw Toxic Comment Classification Challenge dataset. Three random
seeds per model. Licensed under Apache License 2.0.

## Verifying the paper's numbers

Every table, figure, and statistic reported in the manuscript traces to a file in this repository.
`TRACEABILITY.md` maps each manuscript claim to its source file, so a reviewer can check any
number without retraining anything. File integrity can be checked against `SHASUMS`
(`sha256sum -c SHASUMS`).

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
