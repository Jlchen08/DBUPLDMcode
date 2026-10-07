# DBUPLDM

Dual Balanced Large Margin Machine with Unified Pinball Loss for Imbalanced Data Classification.

## Layout

- `matlab/` — core model code (DBUPLDM solver `Unified_pin_csldm.m`, LDM/SVM/Pin-SVM variants, kernel and fuzzy-membership utilities, tuning helpers) plus the `main.m` demo entry.
- `python/` — large-scale demonstration on KDD99-10 (`scalability_kdd99.py`): shared random projection plus mini-batch SGD comparing SVM, PinSVM, UPSVM, LDM, CSLDM and DBUPLDM.

## Requirements

- MATLAB (any recent release; Statistics and Machine Learning Toolbox for `xlsread`/`xlswrite` if the `main.m` demo is used).
- Python 3 with `numpy`, `scikit-learn`, `pandas`, `scipy`, `joblib`.

## Usage

MATLAB: place the benchmark data files next to `matlab/main.m` and run it.

Python: `python python/scalability_kdd99.py` downloads KDD99-10 via `sklearn.datasets.fetch_kddcup99` on first run (seed 42).
