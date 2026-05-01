"""
ml_engine.py — SalesCast AI  (Vercel-compatible)
=================================================
Neural Network (MLP) replaced by Moving Average.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# Dataset preparation  (used by LR and DT)
# ─────────────────────────────────────────────
def prepare_dataset(records, feature_cols, target_col,
                    test_size=0.2, random_seed=42):
    df = pd.DataFrame(records)
    X  = df[feature_cols].values.astype(np.float32)
    y  = df[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
    }


# ─────────────────────────────────────────────
# Shared metric helper  (for LR and DT)
# ─────────────────────────────────────────────
def _metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    return {
        "train_pred": train_pred,
        "test_pred":  test_pred,
        "train_r2":   float(r2_score(y_train, train_pred)),
        "test_r2":    float(r2_score(y_test,  test_pred)),
        "mae":        float(mean_absolute_error(y_test, test_pred)),
        "rmse":       float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "accuracy":   float(100 * np.mean(
                          np.abs(y_test - test_pred) <= 0.2 * np.abs(y_test)
                      )),
    }


# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────
def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    m = _metrics(model, X_train, y_train, X_test, y_test)
    m["model"]              = model
    m["feature_importance"] = np.abs(model.coef_)
    return m


# ─────────────────────────────────────────────
# Decision Tree
# ─────────────────────────────────────────────
def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=5):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    m = _metrics(model, X_train, y_train, X_test, y_test)
    m["model"]              = model
    m["feature_importance"] = model.feature_importances_
    return m


# ─────────────────────────────────────────────
# Moving Average
# ─────────────────────────────────────────────
def train_moving_average(records, target_col, date_col=None,
                         windows=None):
    """
    Temporal moving-average forecaster.

    - Sorts records by date (if date_col supplied) or by row order.
    - 80 / 20 temporal split (no shuffling).
    - Tries each window in `windows`; keeps the one with lowest test MAE.
    - Prediction for every test point = last MA value seen in training.
    """
    if windows is None:
        windows = [4, 8, 13, 26, 52]

    df = pd.DataFrame(records)

    # ── Sort by date ──────────────────────────────────────────────────────
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    sales      = df[target_col].values.astype(np.float64)
    train_size = int(0.8 * len(sales))
    train_s    = sales[:train_size]
    test_s     = sales[train_size:]

    # ── Find best window ─────────────────────────────────────────────────
    best_window, best_mae, best_last_ma = None, float("inf"), None

    for w in windows:
        if w >= train_size:
            continue
        ma_series = pd.Series(train_s).rolling(window=w).mean()
        last_ma   = float(ma_series.iloc[-1])
        if np.isnan(last_ma):
            continue
        preds = np.full(len(test_s), last_ma)
        mae   = float(mean_absolute_error(test_s, preds))
        if mae < best_mae:
            best_mae, best_window, best_last_ma = mae, w, last_ma

    # Fallback if no window worked
    if best_window is None:
        best_window  = windows[0]
        best_last_ma = float(np.mean(train_s))

    # ── Build predictions ─────────────────────────────────────────────────
    # Train: rolling MA with min_periods=1 so no leading NaN
    train_pred = (
        pd.Series(train_s)
        .rolling(window=best_window, min_periods=1)
        .mean()
        .values
    )
    test_pred = np.full(len(test_s), best_last_ma)

    # ── Metrics ───────────────────────────────────────────────────────────
    train_r2 = float(r2_score(train_s, train_pred))
    test_r2  = float(r2_score(test_s,  test_pred))
    mae_val  = float(mean_absolute_error(test_s,  test_pred))
    rmse_val = float(np.sqrt(mean_squared_error(test_s, test_pred)))
    accuracy = float(100 * np.mean(
        np.abs(test_s - test_pred) <= 0.2 * np.abs(test_s)
    ))

    return {
        "train_pred":        train_pred,
        "test_pred":         test_pred,
        "y_train":           train_s,
        "y_test":            test_s,
        "train_r2":          train_r2,
        "test_r2":           test_r2,
        "mae":               mae_val,
        "rmse":              rmse_val,
        "accuracy":          accuracy,
        "best_window":       best_window,
        "last_ma":           best_last_ma,
        # One "feature": the chosen window — used for the importance chart
        "feature_importance": [float(best_window)],
        "feature_names":      [f"MA window = {best_window}"],
    }


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
def run_pipeline(records, feature_cols, target_col,
                 models=None, config=None):
    if models is None:
        models = ["lr", "dt", "ma"]
    if config is None:
        config = {}

    results       = {}
    model_objects = {}

    # ── Prepare dataset for LR / DT  ─────────────────────────────────────
    needs_ds = any(k in models for k in ("lr", "dt"))
    ds = None
    if needs_ds:
        ds = prepare_dataset(
            records, feature_cols, target_col,
            test_size=config.get("test_split",   0.2),
            random_seed=config.get("random_seed", 42),
        )

    # ── Linear Regression ─────────────────────────────────────────────────
    if "lr" in models:
        try:
            lr = train_linear_regression(
                ds["X_train"], ds["y_train"],
                ds["X_test"],  ds["y_test"],
            )
            model_objects["lr"] = {"model": lr["model"], "scaler": ds["scaler"]}
            results["lr"] = {
                "name":  "Linear Regression",
                "icon":  "📈",
                "color": "#4f8ef7",
                "train": {"r2": lr["train_r2"]},
                "test":  {"r2": lr["test_r2"], "mae": lr["mae"],
                          "rmse": lr["rmse"], "acc": lr["accuracy"]},
                "predictions":        lr["test_pred"].tolist(),
                "actuals":            ds["y_test"].tolist(),
                "feature_importance": lr["feature_importance"].tolist(),
                "feature_names":      feature_cols,
            }
        except Exception as e:
            results["lr"] = {"error": f"Linear Regression failed: {e}"}

    # ── Decision Tree ──────────────────────────────────────────────────────
    if "dt" in models:
        try:
            dt = train_decision_tree(
                ds["X_train"], ds["y_train"],
                ds["X_test"],  ds["y_test"],
                max_depth=config.get("max_depth", 5),
            )
            model_objects["dt"] = {"model": dt["model"], "scaler": ds["scaler"]}
            results["dt"] = {
                "name":  "Decision Tree",
                "icon":  "🌳",
                "color": "#00d4aa",
                "train": {"r2": dt["train_r2"]},
                "test":  {"r2": dt["test_r2"], "mae": dt["mae"],
                          "rmse": dt["rmse"], "acc": dt["accuracy"]},
                "predictions":        dt["test_pred"].tolist(),
                "actuals":            ds["y_test"].tolist(),
                "feature_importance": dt["feature_importance"].tolist(),
                "feature_names":      feature_cols,
            }
        except Exception as e:
            results["dt"] = {"error": f"Decision Tree failed: {e}"}

    # ── Moving Average ─────────────────────────────────────────────────────
    if "ma" in models:
        try:
            date_col = config.get("date_col")          # optional
            windows  = config.get("ma_windows", [4, 8, 13, 26, 52])

            ma = train_moving_average(
                records, target_col,
                date_col=date_col,
                windows=windows,
            )
            # Store just enough to predict later (last_ma is the predictor)
            model_objects["ma"] = {"last_ma": ma["last_ma"]}

            results["ma"] = {
                "name":  f"Moving Average (w={ma['best_window']})",
                "icon":  "📉",
                "color": "#7c5cfc",
                "train": {"r2": ma["train_r2"]},
                "test":  {"r2": ma["test_r2"], "mae": ma["mae"],
                          "rmse": ma["rmse"],  "acc": ma["accuracy"]},
                "predictions":        ma["test_pred"].tolist(),
                "actuals":            ma["y_test"].tolist(),
                "feature_importance": ma["feature_importance"],
                "feature_names":      ma["feature_names"],
                "best_window":        ma["best_window"],
            }
        except Exception as e:
            results["ma"] = {"error": f"Moving Average failed: {e}"}

    # Derive dataset_info (use MA counts if LR/DT weren't run)
    if ds:
        n_train = len(ds["y_train"])
        n_test  = len(ds["y_test"])
    elif "ma" in results and not results["ma"].get("error"):
        n_test  = len(results["ma"]["actuals"])
        n_train = len(records) - n_test
    else:
        n_train = n_test = 0

    return {
        "models":        results,
        "model_objects": model_objects,
        "dataset_info": {
            "n_samples_total": n_train + n_test,
            "n_train":         n_train,
            "n_test":          n_test,
        },
    }