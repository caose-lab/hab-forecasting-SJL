# Operational Model Registry

This folder contains the recommended artifacts for operational use.

## Week 1

- Regression
  - Bundle: `week1/regression/chl_weekly_forecast_bundle.joblib`
  - Report: `week1/regression/training_report.json`
  - Purpose: predict average `CHLL_NN_TOTAL` for days `t` through `t+6`

- 3-class risk
  - Bundle: `week1/risk_3class/week1_risk_model_bundle.joblib`
  - Report: `week1/risk_3class/week1_risk_training_report.json`
  - Purpose: low / medium / high week-1 risk communication
  - Thresholding: fixed high-risk cutoff at `16.41 mg/m^3`; low-risk cutoff remains the training-set 25th percentile
  - Note: this model is not the preferred bloom alerting model
  - Note: refined multiclass version uses the compact week-1 feature set without `AWND` or `tidal_range`, plus inverse-frequency weighting

- Binary high-risk alert
  - Bundle: `week1/high_risk/week1_high_risk_bundle.joblib`
  - Report: `week1/high_risk/week1_high_risk_training_report.json`
  - Purpose: bloom-alert style `high` vs `not_high` detection
  - Threshold: fixed bloom cutoff at `16.41 mg/m^3`
  - Note: deployed as a recall-oriented alert model using the broader week-2-style classifier feature set and a low probability threshold (`0.10`) to favor detection of high-risk weeks

## Week 2

- Regression
  - Bundle: `week2/regression/chl_weekly_forecast_bundle.joblib`
  - Report: `week2/regression/training_report.json`
  - Purpose: predict average `CHLL_NN_TOTAL` for days `t+7` through `t+13`
  - Environmental inputs: `precipitation`, `air_temperature`, `water_temperature`, `water_level`, `Watt_per_m2`, `AWND`, and `tidal_range`

- 3-class risk
  - Bundle: `week2/risk_3class/horizon_2_risk_model_bundle.joblib`
  - Report: `week2/risk_3class/horizon_2_risk_training_report.json`
  - Purpose: low / medium / high week-2 risk communication
  - Thresholding: fixed high-risk cutoff at `16.41 mg/m^3`; low-risk cutoff remains the training-set 25th percentile
  - Note: refined multiclass version uses the broader pruned week-3-style feature pool with square-root inverse-frequency weighting

- Binary high-risk alert
  - Bundle: `week2/high_risk/horizon_2_high_risk_bundle.joblib`
  - Report: `week2/high_risk/horizon_2_high_risk_training_report.json`
  - Purpose: bloom-alert style `high` vs `not_high` detection
  - Threshold: fixed bloom cutoff at `16.41 mg/m^3`
  - Note: deployed as a recall-oriented alert model with `AWND` retained and a tuned probability threshold (`0.125`)

## Week 3

- Regression
  - Bundle: `week3/regression/chl_weekly_forecast_bundle.joblib`
  - Report: `week3/regression/training_report.json`
  - Purpose: predict average `CHLL_NN_TOTAL` for days `t+14` through `t+20`
  - Environmental inputs: compact broad quality-gated feature set with recent target history, calendar terms, and a reduced engineered predictor pool
  - Note: the deployed week-3 model uses a leaner 241-feature profile; this sacrifices a small amount of accuracy relative to the broader week-3 variant in exchange for a much smaller feature set

- 3-class risk
  - Bundle: `week3/risk_3class/horizon_3_risk_model_bundle.joblib`
  - Report: `week3/risk_3class/horizon_3_risk_training_report.json`
  - Purpose: low / medium / high week-3 risk communication
  - Thresholding: fixed high-risk cutoff at `16.41 mg/m^3`; low-risk cutoff remains the training-set 25th percentile
  - Note: this model uses the broader pruned week-3 feature profile rather than the compact regression profile

- Binary high-risk alert
  - Bundle: `week3/high_risk/horizon_3_high_risk_bundle.joblib`
  - Report: `week3/high_risk/horizon_3_high_risk_training_report.json`
  - Purpose: bloom-alert style `high` vs `not_high` detection at the 3-week horizon
  - Threshold: fixed bloom cutoff at `16.41 mg/m^3`
  - Note: this model uses the broader pruned week-3 feature profile and a recall-oriented threshold (`0.10`) because that substantially improved high-risk detection relative to the prior conservative deployment

## Recommended Use

- Use regression bundles for expected weekly-average magnitude.
- Use 3-class risk bundles for general risk communication.
- Use binary high-risk bundles for alerting workflows.
- Use `scripts/predict_operational_package.py` to emit the combined week-1, week-2, and week-3 regression and risk package in one JSON response.
