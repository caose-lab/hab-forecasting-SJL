# HAB Forecasting Handoff

This repository contains the current operational HAB forecasting workflow for 1-, 2-, and 3-week horizons.

## Core locations

- `src/chl_forecast/forecasting.py`: training logic, feature engineering, model defaults
- `scripts/predict_operational_package.py`: emits the combined operational package
- `operational_models/`: deployed regression, 3-class risk, and binary high-risk bundles
- `diagnostics/`: forecast diagnostics, scatter plots, confusion matrices, fan-chart exports

## Current deployed models

- Regression:
  - Week 1: MAE `2.423`, RMSE `3.208`, R² `0.475`
  - Week 2: MAE `2.647`, RMSE `3.422`, R² `0.402`
  - Week 3: MAE `3.079`, RMSE `3.905`, R² `0.221`
- Binary high-risk alert:
  - Week 1: balanced accuracy `0.732`, precision `0.221`, recall `0.758`, F1 `0.342`, ROC-AUC `0.836`
  - Week 2: balanced accuracy `0.720`, precision `0.286`, recall `0.606`, F1 `0.388`, ROC-AUC `0.835`
  - Week 3: balanced accuracy `0.677`, precision `0.291`, recall `0.485`, F1 `0.364`, ROC-AUC `0.741`
- 3-class risk:
  - Week 1: accuracy `0.634`, balanced accuracy `0.591`, macro F1 `0.567`
  - Week 2: accuracy `0.556`, balanced accuracy `0.505`, macro F1 `0.518`
  - Week 3: accuracy `0.586`, balanced accuracy `0.477`, macro F1 `0.469`

## Important modeling choices

- Fixed bloom threshold: `16.41 mg m^-3`
- Satellite-derived variables are quality-gated when `coverage_percent < 40`
- Operational prediction uses only data through `t-1`
- Regression outputs include split-conformal intervals at 50%, 68%, 80%, and 90%
- `predict_operational_package.py` now adds a combined `operational_assessment` layer

## Recommended next work

- Clean up and document GitHub-facing repo structure further
- Continue systematic search for 3-class week-3 risk if better recall is needed
- Consider adding a concise API/CLI wrapper for operational use
