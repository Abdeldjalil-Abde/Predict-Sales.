"""
ml_engine.py — SalesCast AI  (Vercel-compatible)
=================================================
PyTorch has been replaced with sklearn.neural_network.MLPRegressor.
This cuts the dependency footprint from ~750 MB to ~30 MB and keeps
the function well inside Vercel's 250 MB compressed limit.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# Dataset preparation
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
# Shared metric helper
# ─────────────────────────────────────────────
def _metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    return {
        "train_pred":  train_pred,
        "test_pred":   test_pred,
        "train_r2":    float(r2_score(y_train, train_pred)),
        "test_r2":     float(r2_score(y_test,  test_pred)),
        "mae":         float(mean_absolute_error(y_test, test_pred)),
        "rmse":        float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "accuracy":    float(100 * np.mean(
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
# Neural Network  (sklearn MLP — no PyTorch)
# ─────────────────────────────────────────────
def train_neural_network(X_train, y_train, X_test, y_test,
                         epochs=200, hidden_units=64):
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_units, hidden_units // 2),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X_train, y_train)
    m = _metrics(model, X_train, y_train, X_test, y_test)

    # Approximate feature importance via column-zeroing
    baseline_mse = mean_squared_error(y_test, m["test_pred"])
    importance   = []
    for i in range(X_test.shape[1]):
        X_tmp       = X_test.copy()
        X_tmp[:, i] = 0
        imp = max(0.0, float(mean_squared_error(y_test, model.predict(X_tmp)) - baseline_mse))
        importance.append(imp)

    m["model"]              = model
    m["feature_importance"] = importance
    return m


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
def run_pipeline(records, feature_cols, target_col,
                 models=None, config=None):
    if models is None:
        models = ["lr", "dt", "nn"]
    if config is None:
        config = {}

    ds = prepare_dataset(
        records, feature_cols, target_col,
        test_size=config.get("test_split",  0.2),
        random_seed=config.get("random_seed", 42),
    )

    results = {}
    model_objects = {}   # kept separate — not JSON-serialisable

    # ── Linear Regression ────────────────────
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

    # ── Decision Tree ─────────────────────────
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

    # ── Neural Network (MLP) ──────────────────
    if "nn" in models:
        try:
            nn = train_neural_network(
                ds["X_train"], ds["y_train"],
                ds["X_test"],  ds["y_test"],
                epochs=config.get("nn_epochs", 200),
                hidden_units=config.get("nn_hidden_units", 64),
            )
            model_objects["nn"] = {"model": nn["model"], "scaler": ds["scaler"]}
            results["nn"] = {
                "name":  "Neural Network (MLP)",
                "icon":  "🧠",
                "color": "#7c5cfc",
                "train": {"r2": nn["train_r2"]},
                "test":  {"r2": nn["test_r2"], "mae": nn["mae"],
                          "rmse": nn["rmse"], "acc": nn["accuracy"]},
                "predictions":        nn["test_pred"].tolist(),
                "actuals":            ds["y_test"].tolist(),
                "feature_importance": nn["feature_importance"],
                "feature_names":      feature_cols,
            }
        except Exception as e:
            results["nn"] = {"error": f"Neural Network failed: {e}"}

    return {
        "models":       results,
        "model_objects": model_objects,    # passed back to app.py for /tmp storage
        "dataset_info": {
            "n_samples_total": len(ds["y_train"]) + len(ds["y_test"]),
            "n_train":         len(ds["y_train"]),
            "n_test":          len(ds["y_test"]),
        },
    }
