# NYC 311: predicting time-to-close

Predict how many hours an NYC 311 service request will take to close, using only
information available when the ticket is created.

## Data

A 300,000-row sample of the NYC Open Data **311 Service Requests** dataset
(tickets created Jan 2024 to Nov 2025). The CSV is not included in this repo.
Download the dataset from [NYC Open Data](https://data.cityofnewyork.us/) and save a sample
as `nyc311_sample.csv` next to the script.

## Approach

| Step | Choice | Why |
|---|---|---|
| Target | `log1p(hours_to_close)` | Durations are heavily skewed (median ~5 h, max ~22,000 h). |
| Cleaning | Dropped open tickets and negative durations | No valid label. |
| Leakage | Dropped `status`, `resolution_description`, `resolution_action_updated_date`, `closed_date` | Only known after the ticket is resolved. |
| Split | By `created_date`: 70% train, 15% val, 15% test | Random splits would let the model see the future. |
| Categoricals | Top 200 values per column, rest grouped as `Other`; one-hot encoded | Limits width and rare-category noise. Top-N is learned on train only. |
| Missing values | Median imputation + scaling for numerics inside the pipeline | Needed by the linear model, no leakage because it is fit on train only. |
| Models | Linear, XGBoost, CatBoost, LightGBM | Compared untuned on the validation set. |
| Tuning | Optuna (30 trials) on LightGBM | Cheaper than a grid search. |

Features: agency, complaint type, descriptor, location type, address type, community board,
police precinct, borough, channel, latitude/longitude, council district, and the hour,
weekday, month and weekend flag of `created_date`.

## Results

Untuned models, validation set (RMSE on `log1p(hours)`):

| Model | RMSE | R² |
|---|---|---|
| XGBoost | 1.041 | 0.798 |
| LightGBM | 1.043 | 0.798 |
| CatBoost | 1.049 | 0.795 |
| Linear | 1.068 | 0.788 |

After Optuna tuning, LightGBM reaches a validation RMSE of 1.032.

**Final test set** (newest 15% of tickets, never used for tuning):

| RMSE (log) | R² (log) | MAE (log) |
|---|---|---|
| 1.146 | 0.739 | 0.789 |

Test error by true ticket duration (error is mean absolute error on the log scale):

| Bucket | Tickets | Median true (h) | Median predicted (h) | Mean error |
|---|---|---|---|---|
| <1h | 12,655 | 0.43 | 1.48 | 0.85 |
| 1-6h | 11,197 | 2.12 | 1.95 | 0.47 |
| 6-24h | 4,846 | 12.90 | 18.62 | 1.01 |
| 1-3d | 5,032 | 44.75 | 39.83 | 0.72 |
| 3-7d | 3,509 | 105.76 | 109.68 | 0.77 |
| 1-4wk | 4,235 | 306.65 | 195.99 | 0.78 |
| >4wk | 2,753 | 1492.25 | 315.63 | 1.62 |

## Findings

- **Agency explains most of the signal.** It carries roughly 60 to 70% of the importance in
  the tree models, followed by complaint type, descriptor and location type.
- **The model is accurate for typical tickets and weak in the tails.** Mid-range tickets are
  predicted close to their true median. Tickets that take over four weeks are underpredicted
  by about 5x, and some very fast tickets are overpredicted.
- **The models are essentially tied.** XGBoost, LightGBM and CatBoost are within about 0.01
  RMSE of each other. Tuning gained about 0.01 RMSE.
- **Extra features did not help.** Agency x complaint combinations, a rolling 24-hour agency
  workload count, business-hours and day-of-month features were tested. Validation RMSE
  was 1.0328 with them and 1.0318 without, and test RMSE was 1.141 with them and 1.146
  without. The differences are within noise, so they were left out.
- **Fast DOB tickets are a failure case.** Tickets closed in under an hour have a
  mean log error of about 4.6 for DOB, versus about 0.55 for NYPD.

## Caveats

- **Censoring bias.** Tickets that are still open are dropped, which removes the slowest
  tickets, mostly from the newest period. The test set therefore looks faster than reality.
- **Temporal drift.** Test error is higher than validation error because the test period is
  the furthest from training. This is expected with a time-based split.
- **Noise ceiling.** Two similar tickets can close weeks apart for reasons not in the data,
  so there is a limit to how good a model on these columns can be.
- **Validation was used for tuning.** Validation scores are optimistic. Use the test numbers.
- **Metrics are on the log scale.** A prediction from a log model approximates the median, not
  the mean, so hour-scale errors are better judged with medians than RMSE.

## Run it

```bash
pip install -r requirements.txt
python nyc311_hours_to_close.py
```

Set `N_TRIALS = 3` at the top of the script for a quick smoke test. Plots and the fitted
model are written to `outputs/`.

## Ideas for next steps

- Handle censoring by cutting the data at a date and keeping only tickets old enough to have closed.
- Use native categorical handling in LightGBM or CatBoost, or target encoding.
- Frame it as a classification (fast, medium, slow) or predict ranges with quantile regression.
- Retrain on train + validation before using the model on new data.
