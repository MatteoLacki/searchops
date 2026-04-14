import pandas as pd
import re
import numpy as np
import xgboost
import matplotlib.pyplot as plt

from kilograms import scatterplot_matrix
from pandas_ops.io import read_df
from timstofu.conversion.reformat import open_pmsms
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)
sagefdr = read_df(
    "temp/F9477/optimal2tier4/sage/devel_fixed/p12f15/human/results/results.sage.fdr.parquet"
)

# sagefdr["logprecmz"] = np.log10(sagefdr.calcmass)

# scatterplot_matrix(
#     sagefdr[["calcmass", "expmass", "precursor_ppm", "fragment_ppm", "rt"]]
# )

sagefdr.fragment_ppm.min()
sagefdr.fragment_ppm.max()
sagefdr.precursor_ppm.min()
sagefdr.precursor_ppm.max()

# ── XGBoost: precursor_ppm ~ expmass + rt ────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

X = sagefdr[["expmass", "rt"]].values
X = sagefdr[["expmass"]].values
y = sagefdr["precursor_ppm"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgboost.XGBRegressor(
    objective="reg:absoluteerror",
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

y_pred = model.predict(X_test)
print(f"MAE  : {np.mean(np.abs(y_test - y_pred)):.4f} ppm")
print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.4f} ppm")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
# print(f"Feature importances: expmass={model.feature_importances_[0]:.3f}  rt={model.feature_importances_[1]:.3f}")


# plt.scatter(X, y, s=1, alpha=0.5)
# plt.scatter(X_test, y_pred, c="black")
# plt.show()

# ── XGBoost: fragment_ppm ~ expmass ──────────────────────────────────────────
X_frag = sagefdr[["expmass"]].values
y_frag = sagefdr["fragment_ppm"].values

X_frag_train, X_frag_test, y_frag_train, y_frag_test = train_test_split(
    X_frag, y_frag, test_size=0.2, random_state=42
)

model_frag = xgboost.XGBRegressor(
    objective="reg:absoluteerror",
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model_frag.fit(
    X_frag_train, y_frag_train, eval_set=[(X_frag_test, y_frag_test)], verbose=50
)

y_frag_pred = model_frag.predict(X_frag_test)
print(f"MAE  : {np.mean(np.abs(y_frag_test - y_frag_pred)):.4f} ppm")
print(f"RMSE : {root_mean_squared_error(y_frag_test, y_frag_pred):.4f} ppm")
print(f"R²   : {r2_score(y_frag_test, y_frag_pred):.4f}")

# plt.scatter(X_frag, y_frag, s=1, alpha=0.5)
# plt.scatter(X_frag_test, y_frag_pred, c="black")
# plt.show()


pipe = make_pipeline(
    SplineTransformer(
        degree=2,
        n_knots=10,
        knots="uniform",
        include_bias=False,
        extrapolation="constant",
    ),
    StandardScaler(with_mean=False),  # keep the intercept separate
    BayesianRidge(),  # fit_intercept=True (default)
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipe.fit(X_train, y_train)
# Predictions on the test set (or on the whole grid for plotting)
y_pred = pipe.predict(X_test)

print(f"Test R² = {model.score(X_test, y_test):.3f}")


# ── heteroscedastic model: fit |residuals| ~ expmass ─────────────────────────
pipe_var = make_pipeline(
    SplineTransformer(degree=2, n_knots=10, knots="uniform", include_bias=False, extrapolation="constant"),
    StandardScaler(with_mean=False),
    BayesianRidge(),
)
abs_resid_train = np.abs(y_train - pipe.predict(X_train))
pipe_var.fit(X_train, abs_resid_train)

# ── plot: data + mean + ±2σ heteroscedastic band ─────────────────────────────
x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 500).reshape(-1, 1)
y_grid_mean = pipe.predict(x_grid)
y_grid_std = np.clip(pipe_var.predict(x_grid), 0, None)

plt.scatter(X, y, s=1, alpha=0.3, label="data")
plt.plot(x_grid, y_grid_mean, c="black", lw=1.5, label="mean")
plt.fill_between(
    x_grid[:, 0],
    y_grid_mean - 2 * y_grid_std,
    y_grid_mean + 2 * y_grid_std,
    alpha=0.3,
    label="±2σ",
)
plt.xlabel("expmass")
plt.ylabel("precursor_ppm")
plt.legend()
plt.show()
