# Release / Handoff Note

This repository includes the current versioned `.joblib` model bundles used by the operational HAB forecasting workflow.

## Where The Bundles Live

Primary deployed artifacts:

- `operational_models/week1/regression/chl_weekly_forecast_bundle.joblib`
- `operational_models/week1/risk_3class/week1_risk_model_bundle.joblib`
- `operational_models/week1/high_risk/week1_high_risk_bundle.joblib`
- `operational_models/week2/regression/chl_weekly_forecast_bundle.joblib`
- `operational_models/week2/risk_3class/horizon_2_risk_model_bundle.joblib`
- `operational_models/week2/high_risk/horizon_2_high_risk_bundle.joblib`
- `operational_models/week3/regression/chl_weekly_forecast_bundle.joblib`
- `operational_models/week3/risk_3class/horizon_3_risk_model_bundle.joblib`
- `operational_models/week3/high_risk/horizon_3_high_risk_bundle.joblib`

Legacy consolidated training artifact:

- `models/chl_weekly_forecast_bundle.joblib`

## How To Regenerate Them

All training commands assume:

```bash
export PYTHONPATH=$PWD/src
```

### Regression

```bash
python3 scripts/train_model.py \
  --csv /path/to/SJL_daily_df.csv \
  --output-dir models
```

### 3-class Risk

```bash
python3 scripts/train_week1_risk_model.py \
  --csv /path/to/SJL_daily_df.csv \
  --output-dir operational_models/week1/risk_3class \
  --fixed-high-threshold 16.41
```

```bash
python3 - <<'PY'
from chl_forecast.forecasting import train_horizon_risk_model
train_horizon_risk_model('/path/to/SJL_daily_df.csv', 'operational_models/week2/risk_3class', horizon=2, fixed_high_threshold=16.41)
train_horizon_risk_model('/path/to/SJL_daily_df.csv', 'operational_models/week3/risk_3class', horizon=3, fixed_high_threshold=16.41)
PY
```

### Binary High-Risk Alert

```bash
python3 scripts/train_week1_high_risk_model.py \
  --csv /path/to/SJL_daily_df.csv \
  --output-dir operational_models/week1/high_risk \
  --fixed-high-threshold 16.41
```

```bash
python3 scripts/train_horizon_high_risk_model.py \
  --csv /path/to/SJL_daily_df.csv \
  --horizon 2 \
  --output-dir operational_models/week2/high_risk \
  --fixed-high-threshold 16.41
```

```bash
python3 scripts/train_horizon_high_risk_model.py \
  --csv /path/to/SJL_daily_df.csv \
  --horizon 3 \
  --output-dir operational_models/week3/high_risk \
  --fixed-high-threshold 16.41
```

## Operational Forecast Command

```bash
python3 scripts/predict_operational_package.py \
  --csv /path/to/SJL_daily_df.csv \
  --prediction-date 2026-03-17
```

## Notes

- The repository includes the current model bundles for convenience and continuity of handoff.
- The authoritative description of intended operational use is in `operational_models/MODEL_REGISTRY.md`.
- If bundle files are ever removed from version control in the future, this note should be retained so the artifacts can be rebuilt deterministically.
