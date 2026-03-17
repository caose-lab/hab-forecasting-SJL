# HAB Forecasting for San Juan Lagoon (SJL)

Operational forecasting system for weekly chlorophyll-based harmful algal bloom (HAB) risk in San Juan Lagoon. The repository contains the current forecasting codebase, deployed model artifacts, diagnostics, and handoff documentation needed to continue development.

## What This Repository Contains

- 1-, 2-, and 3-week regression forecasts of weekly mean `CHLL_NN_TOTAL`
- 3-class risk models (`low`, `medium`, `high`) for weeks 1-3
- binary high-risk bloom-alert models for weeks 1-3
- conformal prediction intervals for regression forecasts
- an operational decision layer that combines regression, binary alerts, and 3-class risk outputs
- diagnostics, holdout predictions, and publication-ready figures

## Repository Structure

- `src/chl_forecast/forecasting.py`
  Core training, feature engineering, evaluation, and inference logic.
- `scripts/`
  Command-line entry points for training, prediction, search, diagnostics, and figure generation.
- `operational_models/`
  Deployed week-1, week-2, and week-3 regression and classification bundles plus reports.
- `diagnostics/`
  Regression and classification diagnostics, scatter plots, confusion matrices, and fan-chart exports.
- `HANDOFF.md`
  High-level project status, deployed metrics, and next development directions.
- `RUNBOOK.md`
  Reproducible commands for forecasting, retraining, and regenerating diagnostics.
- `RELEASE_HANDOFF.md`
  Notes specific to the versioned model bundles and how to rebuild them.

## Current Operational Models

### Regression

- Week 1: MAE `2.423`, RMSE `3.208`, R² `0.475`
- Week 2: MAE `2.647`, RMSE `3.422`, R² `0.402`
- Week 3: MAE `3.079`, RMSE `3.905`, R² `0.221`

### Binary High-Risk Alert

- Week 1: balanced accuracy `0.732`, precision `0.221`, recall `0.758`, F1 `0.342`, ROC-AUC `0.836`
- Week 2: balanced accuracy `0.720`, precision `0.286`, recall `0.606`, F1 `0.388`, ROC-AUC `0.835`
- Week 3: balanced accuracy `0.677`, precision `0.291`, recall `0.485`, F1 `0.364`, ROC-AUC `0.741`

### 3-Class Risk

- Week 1: accuracy `0.634`, balanced accuracy `0.591`, macro F1 `0.567`
- Week 2: accuracy `0.556`, balanced accuracy `0.505`, macro F1 `0.518`
- Week 3: accuracy `0.586`, balanced accuracy `0.477`, macro F1 `0.469`

## Key Modeling Choices

- Fixed bloom threshold: `16.41 mg m^-3`
- Satellite-derived variables are quality-gated when `coverage_percent < 40`
- All operational predictors use only information available through `t-1`
- Regression forecasts include split-conformal intervals at 50%, 68%, 80%, and 90%
- `scripts/predict_operational_package.py` emits a combined operational package with an `operational_assessment` summary

## Quick Start

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

Generate the combined operational forecast package:

```bash
PYTHONPATH=$PWD/src python3 scripts/predict_operational_package.py \
  --csv /path/to/SJL_daily_df.csv \
  --prediction-date 2026-03-17
```

## Model Bundles

This repository includes the current `.joblib` bundles used in operations under:

- `models/`
- `operational_models/week1/`
- `operational_models/week2/`
- `operational_models/week3/`

For usage guidance by model type, see `operational_models/MODEL_REGISTRY.md`.

## Development Workflow

Useful entry points:

- `scripts/train_model.py`
- `scripts/train_week1_risk_model.py`
- `scripts/train_horizon_risk_model.py`
- `scripts/train_week1_high_risk_model.py`
- `scripts/train_horizon_high_risk_model.py`
- `scripts/search_week1_regression.py`
- `scripts/search_week1_regression_focused.py`
- `scripts/search_binary_high_risk_models.py`
- `scripts/search_multiclass_risk_models.py`

## Diagnostics and Figures

This repository also includes:

- regression holdout diagnostics and persistence comparisons
- binary high-risk confusion matrices
- fan-chart CSV exports for MATLAB
- publication-ready regression comparison figures

## Handoff

If you are taking over development, start with:

1. `HANDOFF.md`
2. `RUNBOOK.md`
3. `RELEASE_HANDOFF.md`
4. `operational_models/MODEL_REGISTRY.md`

These documents summarize the deployed models, current metrics, operational logic, and exact commands for reproducing the current system.
