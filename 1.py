import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import optuna
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS = 42
N_TRIALS = 30          # set to 3 for a quick smoke test
TOP_N_CATS = 200

# Untuned params for the initial model comparison.
xgb_params = {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 6,
              'random_state': RS, 'n_jobs': -1, 'objective': 'reg:squarederror'}
lgbm_params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
               'random_state': RS, 'n_jobs': -1, 'verbose': -1, 'objective': 'regression'}
cat_params = {'iterations': 500, 'learning_rate': 0.1, 'depth': 6,
              'random_state': RS, 'verbose': False, 'loss_function': 'RMSE',
              'allow_writing_files': False}

# load + clean
df = pd.read_csv("nyc311_sample.csv")

drop_stage1 = [
    # structurally missing / niche (>90% null)
    'facility_type', 'due_date', 'taxi_company_borough', 'taxi_pick_up_location',
    'bridge_highway_name', 'bridge_highway_direction', 'road_ramp',
    'bridge_highway_segment', 'vehicle_type',
    # redundant copies of other columns
    'park_borough', 'location', 'x_coordinate_state_plane',
    'y_coordinate_state_plane', 'agency_name', 'incident_zip', 'city',
    # high-cardinality or heavily missing, low value
    'intersection_street_1', 'intersection_street_2',
    'cross_street_1', 'cross_street_2', 'landmark', 'descriptor_2',
    'park_facility_name', 'bbl',
    # leakage (known only after resolution)
    'resolution_action_updated_date',
]
df_clean = df.drop(columns=drop_stage1)

cat_cols = ['agency', 'complaint_type', 'descriptor', 'location_type', 'address_type',
            'community_board', 'police_precinct', 'borough', 'open_data_channel_type']
num_cols = ['latitude', 'longitude', 'council_district',
            'hour', 'dayofweek', 'month', 'is_weekend']

df_clean["created_date"] = pd.to_datetime(df_clean["created_date"])
df_clean["closed_date"] = pd.to_datetime(df_clean["closed_date"])

df_clean[["latitude", "longitude", "council_district"]] = (
    df_clean[["latitude", "longitude", "council_district"]].astype("float32")
)

# target
df_clean["hours_to_close"] = (
    (df_clean["closed_date"] - df_clean["created_date"]).dt.total_seconds() / 3600
)

# drop unlabeled (NaN) and negative durations
df_model = df_clean[df_clean["hours_to_close"] >= 0].copy()

# sort by time (the rolling feature and the date split both need this)
df_model = df_model.sort_values("created_date").reset_index(drop=True)

# features
# time features
df_model["hour"] = df_model["created_date"].dt.hour
df_model["dayofweek"] = df_model["created_date"].dt.dayofweek
df_model["month"] = df_model["created_date"].dt.month
df_model["day"] = df_model["created_date"].dt.day
df_model["is_weekend"] = (df_model["dayofweek"] >= 5).astype(int)
df_model["is_business_hours"] = (
    (df_model["hour"].between(9, 17)) & (df_model["dayofweek"] < 5)
).astype(int)

# combined categories
df_model["agency_complaint"] = df_model["agency"].astype(str) + "_" + df_model["complaint_type"].astype(str)
df_model["complaint_descriptor"] = df_model["complaint_type"].astype(str) + "_" + df_model["descriptor"].astype(str)

# clean near-duplicate location types ("Street" vs "STREET ")
df_model["location_type"] = df_model["location_type"].str.upper().str.strip()

# agency workload: tickets that agency received in the previous 24h (past only)
tmp = df_model.set_index("created_date")
df_model["agency_load_24h"] = (
    tmp.groupby("agency")["unique_key"]
       .transform(lambda s: s.rolling("24h").count())
       .values
)

# cat_cols += ["agency_complaint", "complaint_descriptor"]
# num_cols += ["agency_load_24h", "day", "is_business_hours"]

# split by date
c1 = df_model["created_date"].quantile(0.70)
c2 = df_model["created_date"].quantile(0.85)

train = df_model[df_model["created_date"] <= c1]
val = df_model[(df_model["created_date"] > c1) & (df_model["created_date"] <= c2)]
test = df_model[df_model["created_date"] > c2]
print(len(train), len(val), len(test), flush=True)

drop_stage2 = ['closed_date', 'status', 'resolution_description', 'unique_key',
                            'incident_address', 'street_name', 'created_date']
train = train.drop(columns=drop_stage2)
val = val.drop(columns=drop_stage2)
test = test.drop(columns=drop_stage2)

# top-N categories learned from train only; the rest -> "Other"
for c in cat_cols:
    keep = train[c].astype("object").fillna("Unknown").value_counts().nlargest(TOP_N_CATS).index
    for d in (train, val, test):
        d[c] = d[c].astype("object").fillna("Unknown")
        d[c] = d[c].where(d[c].isin(keep), "Other")

X_train, y_train = train[cat_cols + num_cols], np.log1p(train["hours_to_close"])
X_val, y_val = val[cat_cols + num_cols], np.log1p(val["hours_to_close"])
X_test, y_test = test[cat_cols + num_cols], np.log1p(test["hours_to_close"])

# pipeline
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
])


def make_pipeline(model):
    # clone => every pipeline gets its own fresh preprocessor
    return Pipeline([("preprocess", clone(preprocessor)), ("model", model)])


def error_by_bucket(y_true_log, y_pred_log):
    """Error table with human-readable ticket-duration buckets."""
    bins = [0, 1, 6, 24, 72, 168, 720, np.inf]
    labels = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "1-4wk", ">4wk"]
    true_h, pred_h = np.expm1(y_true_log), np.expm1(y_pred_log)
    r = pd.DataFrame({
        "bucket": pd.cut(true_h, bins=bins, labels=labels, include_lowest=True),
        "true_h": true_h, "pred_h": pred_h,
        "abs_error_log": (y_true_log - y_pred_log).abs(),
    })
    return r.groupby("bucket", observed=False).agg(
        tickets=("true_h", "size"),
        median_true_h=("true_h", "median"),
        median_pred_h=("pred_h", "median"),
        mean_error_log=("abs_error_log", "mean"),
    ).round(2)


# baseline comparison
models = {
    "linear": LinearRegression(n_jobs=-1),
    "xgb": XGBRegressor(**xgb_params),
    "catboost": CatBoostRegressor(**cat_params),
    "lgbm": LGBMRegressor(**lgbm_params),
}

trained_models = {}
val_results = {}
for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    p = np.clip(pipe.predict(X_val), 0, None)
    val_results[name] = {
        "rmse": root_mean_squared_error(y_val, p),
        "r2": r2_score(y_val, p),
        "mae": mean_absolute_error(y_val, p),
    }

print("\n" + "=" * 50)
print("VAL RESULTS (untuned)  | old LightGBM baseline was 1.0409")
print("=" * 50)
print(pd.DataFrame(val_results).T.sort_values("rmse"), flush=True)

# Optuna on LightGBM
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = LGBMRegressor(**params, subsample_freq=1, verbose=-1,
                            random_state=RS, n_jobs=-1)
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    return root_mean_squared_error(y_val, pipe.predict(X_val))


study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=RS))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nbest val rmse (Optuna):", round(study.best_value, 4))
print("best params:", study.best_params, flush=True)

# final model = tuned LightGBM, fit on train only
winner = "lgbm_optuna"
final_model = make_pipeline(LGBMRegressor(**study.best_params, subsample_freq=1,
                                            verbose=-1, random_state=RS, n_jobs=-1))
final_model.fit(X_train, y_train)

# validation diagnostics
final_val_pred = np.clip(final_model.predict(X_val), 0, None)
print("\nVAL error by ticket duration (note: val was used for tuning, so it's optimistic)")
print(error_by_bucket(y_val, final_val_pred))

plt.figure(figsize=(10, 6))
plt.scatter(final_val_pred, y_val - final_val_pred, alpha=0.3, s=5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title(f"Residuals vs Predicted (val) - {winner}")
plt.xlabel("Predicted log1p(hours_to_close)")
plt.ylabel("Residual (true - predicted)")
plt.tight_layout()
plt.savefig(f"{winner}_val_residuals_vs_predicted.png")
plt.close()

# final test
test_pred = np.clip(final_model.predict(X_test), 0, None)

print("\n" + "=" * 50)
print(f"FINAL TEST RESULTS - {winner}  | old test RMSE was 1.1538")
print("=" * 50)
print(f"RMSE: {root_mean_squared_error(y_test, test_pred):.4f}")
print(f"R2:   {r2_score(y_test, test_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, test_pred):.4f}")
print(error_by_bucket(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.scatter(test_pred, y_test - test_pred, alpha=0.3, s=5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title(f"Residuals vs Predicted (test) - {winner}")
plt.xlabel("Predicted log1p(hours_to_close)")
plt.ylabel("Residual (true - predicted)")
plt.tight_layout()
plt.savefig(f"{winner}_test_residuals_vs_predicted.png")
plt.close()

# feature importance (untuned models)
for name, pipe in trained_models.items():
    fitted = pipe.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        imp = pd.Series(fitted.feature_importances_, index=names).sort_values(ascending=False)
        print("\n" + "=" * 50)
        print(f"FEATURE IMPORTANCE - {name} (top 15)")
        print("=" * 50)
        print(imp.head(15))

# error slice: fast tickets by agency
res = pd.DataFrame({"agency": X_test["agency"], "true_h": np.expm1(y_test),
                    "err": (y_test - test_pred).abs()})
fast = res[res["true_h"] < 1]
print("\nTickets closed in <1h, by agency:")
print(fast.groupby("agency")["err"].agg(["mean", "count"])
            .sort_values("count", ascending=False).head(8))

# save
out_path = f"nyc311_hours_to_close_{winner}_model.pkl"
joblib.dump({"model": final_model, "winner": winner,
                "best_params": study.best_params,
                "target": "log1p(hours_to_close)"}, out_path)
print(f"\nSaved {out_path}")
