# CHLL_NN_TOTAL Weekly Forecasting

This project trains direct regression models that forecast the average `CHLL_NN_TOTAL` for:

- days `t` through `t+6`
- days `t+7` through `t+13`
- days `t+14` through `t+20`

The models use only information available through the prior day. A prediction row dated `t` is built from observations up to `t-1`, so it matches your operational setup.

Rows with `coverage_percent < 40` are quality-gated before training and inference. On those days, satellite-derived values including `CHLL_NN_TOTAL` are treated as missing and are not used directly by the model.

## Files

- `requirements.txt`: Python dependencies.
- `src/chl_forecast/forecasting.py`: feature engineering, model training, evaluation, and inference logic.
- `scripts/train_model.py`: CLI entry point to fit the models.
- `scripts/predict_latest.py`: CLI entry point to make forecasts from the trained bundle.
- `scripts/backtest_week1.py`: CLI entry point for expanding-window backtests.
- `scripts/search_week1_regression.py`: systematic week-1 regression search across feature sets, transforms, and model families.
- `scripts/train_horizon_risk_model.py`: generic horizon-specific 3-class risk trainer.
- `scripts/predict_horizon_risk.py`: generic horizon-specific 3-class risk predictor.
- `scripts/train_horizon_high_risk_model.py`: generic horizon-specific binary high-risk trainer.
- `scripts/predict_horizon_high_risk.py`: generic horizon-specific binary high-risk predictor.
- `operational_models/MODEL_REGISTRY.md`: curated list of recommended week-1 and week-2 artifacts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Train

```bash
python3 scripts/train_model.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --output-dir models
```

Training writes:

- `models/chl_weekly_forecast_bundle.joblib`
- `models/training_report.json`
- `models/feature_importance_week_1.csv`
- `models/feature_importance_week_2.csv`
- `models/feature_importance_week_3.csv`

## Predict

Predict for the day after the latest date in the CSV:

```bash
python3 scripts/predict_latest.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --bundle models/chl_weekly_forecast_bundle.joblib
```

Use a specific as-of date:

```bash
python3 scripts/predict_latest.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --bundle models/chl_weekly_forecast_bundle.joblib \
  --prediction-date 2026-03-11
```

## Operational Package

Recommended curated artifacts live under `operational_models/`.

Week 1:

- Regression: `operational_models/week1/regression/chl_weekly_forecast_bundle.joblib`
- 3-class risk: `operational_models/week1/risk_3class/week1_risk_model_bundle.joblib`
- Binary high-risk alert: `operational_models/week1/high_risk/week1_high_risk_bundle.joblib`

Week 2:

- Regression: `operational_models/week2/regression/chl_weekly_forecast_bundle.joblib`
- 3-class risk: `operational_models/week2/risk_3class/horizon_2_risk_model_bundle.joblib`
- Binary high-risk alert: `operational_models/week2/high_risk/horizon_2_high_risk_bundle.joblib`

See `operational_models/MODEL_REGISTRY.md` for the recommended use of each artifact.

## Backtest

Run an expanding-window backtest for week 1:

```bash
python3 scripts/backtest_week1.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --horizon 1 \
  --folds 5 \
  --min-train-rows 600
```

## Week 1 Search

Run the systematic week-1 regression search:

```bash
python3 scripts/search_week1_regression.py \
  --csv /Users/a1amador/Downloads/SJL_daily_df.csv \
  --output-dir models_week1_systematic_search
```

This writes:

- `models_week1_systematic_search/week1_regression_search_results.csv`
- `models_week1_systematic_search/week1_regression_search_summary.json`
- `models_week1_systematic_search/week1_candidate_rank_1.joblib`

## Deployment Notes

- The training script uses a chronological holdout and `TimeSeriesSplit`, not random splits.
- Missing values are expected; the pipeline imputes them automatically.
- Feature importance files are permutation-based, so they are easier to interpret than raw tree internals.
- Retraining on a schedule is recommended as new labeled target data arrives.
