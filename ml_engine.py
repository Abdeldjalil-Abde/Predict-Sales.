import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────
def prepare_dataset(records, feature_cols, target_col, test_size=0.2, random_seed=42):
    df = pd.DataFrame(records)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }

# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────
def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate R² properly
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Accuracy: percentage of predictions within 20% of actual
    accuracy = 100 * np.mean(np.abs(y_test - test_pred) <= 0.2 * np.abs(y_test))

    feature_importance = np.abs(model.coef_)

    return {
        "model": model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
    }

# ─────────────────────────────────────────────
# Decision Tree Regressor
# ─────────────────────────────────────────────
def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=5):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    accuracy = 100 * np.mean(np.abs(y_test - test_pred) <= 0.2 * np.abs(y_test))

    feature_importance = model.feature_importances_

    return {
        "model": model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
    }

# ─────────────────────────────────────────────
# Torch Wrapper for Neural Network
# ─────────────────────────────────────────────
class TorchWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, hidden_units=64, epochs=150, lr=0.01):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_units // 2, 1)
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(self.device)

        for _ in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X_tensor), y_tensor)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy().flatten()

# ─────────────────────────────────────────────
# Neural Network training
# ─────────────────────────────────────────────
def train_neural_network(X_train, y_train, X_test, y_test, epochs=150, hidden_units=64):
    wrapper = TorchWrapper(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        epochs=epochs
    )

    wrapper.fit(X_train, y_train)

    train_pred = wrapper.predict(X_train)
    test_pred = wrapper.predict(X_test)

    # Calculate metrics properly
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    accuracy = 100 * np.mean(np.abs(y_test - test_pred) <= 0.2 * np.abs(y_test))

    # Feature importance (drop-column effect)
    feature_importance = []
    baseline_error = mean_squared_error(y_test, test_pred)
    
    for i in range(X_test.shape[1]):
        X_tmp = X_test.copy()
        X_tmp[:, i] = 0
        pred = wrapper.predict(X_tmp)
        error_drop = mean_squared_error(y_test, pred)
        importance = max(0, error_drop - baseline_error)
        feature_importance.append(float(importance))

    return {
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
    }

# ─────────────────────────────────────────────
# Pipeline (CORRECTED VERSION)
# ─────────────────────────────────────────────
def run_pipeline(records, feature_cols, target_col, models=["lr", "dt", "nn"], config={}):
    ds = prepare_dataset(
        records,
        feature_cols,
        target_col,
        test_size=config.get("test_split", 0.2),
        random_seed=config.get("random_seed", 42)
    )

    results = {}

    # =========================
    # Linear Regression
    # =========================
    if "lr" in models:
        try:
            lr = train_linear_regression(
                ds["X_train"], ds["y_train"],
                ds["X_test"], ds["y_test"]
            )

            results["lr"] = {
                "name": "Linear Regression",
                "icon": "📈",
                "color": "#4f8ef7",

                "train": {
                    "r2": float(lr["train_r2"])
                },

                "test": {
                    "r2": float(lr["test_r2"]),
                    "mae": float(lr["mae"]),
                    "rmse": float(lr["rmse"]),
                    "acc": float(lr["accuracy"])
                },

                "predictions": lr["test_pred"].tolist(),
                "actuals": ds["y_test"].tolist(),

                "feature_importance": lr["feature_importance"].tolist(),
                "feature_names": feature_cols
            }
        except Exception as e:
            results["lr"] = {"error": f"Linear Regression failed: {str(e)}"}

    # =========================
    # Decision Tree
    # =========================
    if "dt" in models:
        try:
            dt = train_decision_tree(
                ds["X_train"], ds["y_train"],
                ds["X_test"], ds["y_test"],
                max_depth=config.get("max_depth", 5)
            )

            results["dt"] = {
                "name": "Decision Tree",
                "icon": "🌳",
                "color": "#00d4aa",

                "train": {
                    "r2": float(dt["train_r2"])
                },

                "test": {
                    "r2": float(dt["test_r2"]),
                    "mae": float(dt["mae"]),
                    "rmse": float(dt["rmse"]),
                    "acc": float(dt["accuracy"])
                },

                "predictions": dt["test_pred"].tolist(),
                "actuals": ds["y_test"].tolist(),

                "feature_importance": dt["feature_importance"].tolist(),
                "feature_names": feature_cols
            }
        except Exception as e:
            results["dt"] = {"error": f"Decision Tree failed: {str(e)}"}

    # =========================
    # Neural Network
    # =========================
    if "nn" in models:
        try:
            nn = train_neural_network(
                ds["X_train"], ds["y_train"],
                ds["X_test"], ds["y_test"],
                epochs=config.get("nn_epochs", 150),
                hidden_units=config.get("nn_hidden_units", 64)
            )

            results["nn"] = {
                "name": "Neural Network",
                "icon": "🧠",
                "color": "#7c5cfc",

                "train": {"r2": float(nn["train_r2"])},
                
                "test": {
                    "r2": float(nn["test_r2"]),
                    "mae": float(nn["mae"]),
                    "rmse": float(nn["rmse"]),
                    "acc": float(nn["accuracy"])
                },

                "predictions": nn["test_pred"].tolist(),
                "actuals": ds["y_test"].tolist(),

                "feature_importance": nn["feature_importance"],
                "feature_names": feature_cols
            }
        except Exception as e:
            results["nn"] = {"error": f"Neural Network failed: {str(e)}"}

    return {
        "models": results,
        "dataset_info": {
            "n_samples_total": len(ds["y_train"]) + len(ds["y_test"]),
            "n_train": len(ds["y_train"]),
            "n_test": len(ds["y_test"])
        }
    }
