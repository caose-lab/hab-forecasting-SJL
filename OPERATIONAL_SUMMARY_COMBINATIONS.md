# Operational Summary Combination Matrix

This document describes every operational summary outcome that can be produced by the current decision logic in [`scripts/predict_all_model_outputs.py`](/Users/a1amador/src/hab-forecasting-SJL/scripts/predict_all_model_outputs.py) and [`scripts/predict_operational_package.py`](/Users/a1amador/src/hab-forecasting-SJL/scripts/predict_operational_package.py).

It is intended for frontend implementation so UI states and copy can be mapped correctly.

## Inputs That Drive the Summary

The operational summary is not determined only by these three visible model outputs:

- `HAB alert`: `high` or `not_high`
- `3-class risk`: `low`, `medium`, or `high`
- `Regression forecast`: the point forecast in `mg m^-3`

It also depends on whether the regression forecast or its upper prediction intervals cross the bloom threshold of `16.41 mg m^-3`.

## Regression States Used by the Logic

For frontend purposes, the regression signal should be interpreted as one of these five states:

| Regression state | Meaning |
|---|---|
| `below_all` | Point forecast and upper 50/68/80 intervals are all below `16.41` |
| `upper80_only` | Only the upper 80 interval crosses `16.41` |
| `upper68_80` | Upper 68 and upper 80 intervals cross `16.41` |
| `upper50_68_80` | Upper 50, upper 68, and upper 80 intervals cross `16.41` |
| `point_exceeds` | Point forecast is at or above `16.41` |

## Possible Operational Summary Text

The current logic can emit only these five summary strings:

1. `Low to moderate conditions expected, and elevated bloom risk appears limited.`
2. `Moderate conditions expected, with some uncertainty about elevated bloom risk.`
3. `Moderate conditions expected, but elevated bloom risk remains possible.`
4. `Moderate conditions expected, but elevated bloom risk remains likely.`
5. `Elevated conditions expected, and bloom risk is high.`

## Full Combination Matrix

| HAB alert | 3-class risk | Regression state | Operational summary |
|---|---|---|---|
| `not_high` | `low` | `below_all` | `Low to moderate conditions expected, and elevated bloom risk appears limited.` |
| `not_high` | `low` | `upper80_only` | `Low to moderate conditions expected, and elevated bloom risk appears limited.` |
| `not_high` | `low` | `upper68_80` | `Moderate conditions expected, with some uncertainty about elevated bloom risk.` |
| `not_high` | `low` | `upper50_68_80` | `Moderate conditions expected, with some uncertainty about elevated bloom risk.` |
| `not_high` | `low` | `point_exceeds` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `medium` | `below_all` | `Moderate conditions expected, with some uncertainty about elevated bloom risk.` |
| `not_high` | `medium` | `upper80_only` | `Moderate conditions expected, with some uncertainty about elevated bloom risk.` |
| `not_high` | `medium` | `upper68_80` | `Moderate conditions expected, with some uncertainty about elevated bloom risk.` |
| `not_high` | `medium` | `upper50_68_80` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `medium` | `point_exceeds` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `high` | `below_all` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `high` | `upper80_only` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `high` | `upper68_80` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `high` | `upper50_68_80` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `not_high` | `high` | `point_exceeds` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `high` | `low` | `below_all` | `Low to moderate conditions expected, and elevated bloom risk appears limited.` |
| `high` | `low` | `upper80_only` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `high` | `low` | `upper68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `low` | `upper50_68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `low` | `point_exceeds` | `Elevated conditions expected, and bloom risk is high.` |
| `high` | `medium` | `below_all` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `high` | `medium` | `upper80_only` | `Moderate conditions expected, but elevated bloom risk remains possible.` |
| `high` | `medium` | `upper68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `medium` | `upper50_68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `medium` | `point_exceeds` | `Elevated conditions expected, and bloom risk is high.` |
| `high` | `high` | `below_all` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `high` | `upper80_only` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `high` | `upper68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `high` | `upper50_68_80` | `Moderate conditions expected, but elevated bloom risk remains likely.` |
| `high` | `high` | `point_exceeds` | `Elevated conditions expected, and bloom risk is high.` |

## Important Implementation Notes

- The summary logic uses the bloom threshold `16.41 mg m^-3`.
- The backend also considers probability details that are not visible in the simplified table above.
- A `not_high` HAB alert can still be treated as weak support for elevated risk when its probability is close to the binary threshold.
- A `low` or `medium` 3-class label can still contribute extra support when `prob_high >= 0.30`.
- Because of those probability-based adjustments, a small number of cases can be upgraded by one severity step even when the visible labels look the same.

## Recommended Frontend Mapping

If the frontend only receives final backend outputs, it should display:

- `HAB alert`
- `3-class risk`
- `Regression forecast`
- `Operational summary`

If the frontend needs to reproduce backend behavior locally, it must also receive:

- regression upper intervals at 50%, 68%, and 80%
- binary `prob_high`
- binary `probability_threshold`
- 3-class `prob_high`

