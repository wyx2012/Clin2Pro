# Clin2Pro

Predict longitudinal changes in the plasma proteome from paired clinical measurements.

Clin2Pro uses three training stages: clinical-to-protein cross-sectional pretraining, drug-exposure-associated temporal pretraining, and longitudinal ΔOlink prediction. A pathway projection and a low-rank protein-to-target bridge provide exploratory interpretation of predicted changes.

This repository contains the core training, inference and evaluation implementation. Clinical datasets and trained weights are not included.

## Structure

```text
Clin2Pro/
├── clin2pro/               # Data loaders, models, losses and training stages
├── configs/                  # Full and debug configurations
├── scripts/                  # Evaluation exports and core test runner
├── tests/                    # Core unit tests
├── docs/                     # Input specification and method details
├── run_pipeline.py           # Train the three-stage model
├── predict_new.py            # Predict from paired clinical tables
├── requirements.txt          # Original dependency ranges
├── requirements-tested.txt   # Locally checked dependency versions
├── README.md
└── README.zh-CN.md
```

## Installation

The core tests were checked with Python 3.8.17 and the versions in `requirements-tested.txt`. Run the following in a compatible Python environment, from the repository root:

```bash
python -m pip install -r requirements-tested.txt
python scripts/run_core_tests.py
```

`requirements.txt` preserves the original flexible dependency ranges. Compatibility with other versions, particularly checkpoint loading in newer PyTorch releases, must be checked separately. Load only trusted checkpoints.

## Training

After preparing the data, check the pipeline with the debug configuration, then run full training:

```bash
python run_pipeline.py --config configs/debug.yaml
python run_pipeline.py --config configs/full.yaml
```

To run individual stages, use `--stages 1`, `--stages 2` or `--stages 3`. Stage 2 requires the Stage 1 checkpoint; Stage 3 requires both earlier checkpoints in the same output directory. Debug runs are not manuscript experiments.

## Evaluation

Stage 3 training generates participant-level out-of-fold (OOF) predictions automatically. Additional exports and baseline comparisons are available through:

```bash
python scripts/export_stage1_predictions.py --config configs/full.yaml --oof-folds 5
python scripts/export_stage2_results.py --config configs/full.yaml
python scripts/export_stage1_longitudinal_endpoints.py --config configs/full.yaml
python scripts/export_stage3_benchmarks.py --config configs/full.yaml
```

Stage 1 OOF export trains additional fold models; `--skip-oof` exports only ordinary fitted/validation results. The benchmark script trains comparison models and reuses the existing full-model OOF predictions. See [evaluation details](docs/METHODS.md) for prerequisites and interpretation.

# Clin2Pro
# Clin2Pro
