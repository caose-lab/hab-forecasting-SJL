# Runbook

## Environment

Current commands assume:

- source tree at `/Users/a1amador/Documents/Playground`
- Python env at `/Users/a1amador/miniforge3/envs/habs-codex/bin/python`
- `PYTHONPATH=/Users/a1amador/Documents/Playground/src`

## Operational forecast

```bash
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/predict_operational_package.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --prediction-date 2026-03-17
```

## Retrain deployed binary high-risk models

```bash
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/train_week1_high_risk_model.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --output-dir /Users/a1amador/Documents/Playground/operational_models/week1/high_risk \
  --fixed-high-threshold 16.41
```

```bash
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/train_horizon_high_risk_model.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --horizon 2 \
  --output-dir /Users/a1amador/Documents/Playground/operational_models/week2/high_risk \
  --fixed-high-threshold 16.41
```

```bash
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/train_horizon_high_risk_model.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --horizon 3 \
  --output-dir /Users/a1amador/Documents/Playground/operational_models/week3/high_risk \
  --fixed-high-threshold 16.41
```

## Regenerate figures

Regression diagnostics:

```bash
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/generate_regression_diagnostics.py
```

Binary confusion matrices:

```bash
mkdir -p /tmp/mplconfig && \
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg \
PYTHONPATH=/Users/a1amador/Documents/Playground/src \
/Users/a1amador/miniforge3/envs/habs-codex/bin/python \
/Users/a1amador/Documents/Playground/scripts/assemble_binary_confusion_figure.py
```
