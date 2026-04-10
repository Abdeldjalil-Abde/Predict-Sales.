"""
app.py — SalesCast AI · Flask backend  (Vercel-compatible)
===========================================================
Key changes vs. original
  • PyTorch removed — ml_engine now uses sklearn MLP
  • Trained models pickled to /tmp/salescast_models.pkl so the
    /api/predict endpoint can load them even in a fresh invocation
  • Static files served from the 'static' folder (unchanged)
"""

import os
import json
import pickle
import secrets
import traceback

import numpy as np
import pandas as pd
import requests

from functools import wraps
from urllib.parse import urlencode

from flask import (
    Flask, redirect, request, session,
    url_for, jsonify, send_from_directory,
)

from ml_engine import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

MODELS_PATH = "/tmp/salescast_models.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# OAuth provider config
# ─────────────────────────────────────────────────────────────────────────────
OAUTH = {
    "google": {
        "client_id":     os.environ.get("GOOGLE_CLIENT_ID",     ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        "auth_url":      "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url":     "https://oauth2.googleapis.com/token",
        "userinfo_url":  "https://www.googleapis.com/oauth2/v3/userinfo",
        "scope":         "openid email profile",
    },
    "github": {
        "client_id":     os.environ.get("GITHUB_CLIENT_ID",     ""),
        "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
        "auth_url":      "https://github.com/login/oauth/authorize",
        "token_url":     "https://github.com/login/oauth/access_token",
        "userinfo_url":  "https://api.github.com/user",
        "scope":         "read:user user:email",
    },
    "microsoft": {
        "client_id":     os.environ.get("MICROSOFT_CLIENT_ID",     ""),
        "client_secret": os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
        "auth_url":      "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url":     "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url":  "https://graph.microsoft.com/v1.0/me",
        "scope":         "openid email profile User.Read",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated


def _get_redirect_uri(provider: str) -> str:
    return url_for("oauth_callback", provider=provider, _external=True)


def _json_error(message: str, code: int = 400):
    return jsonify({"status": "error", "message": message}), code


def _save_models(model_objects, feature_cols, target_col):
    """Pickle trained models to /tmp for reuse by /api/predict."""
    try:
        with open(MODELS_PATH, "wb") as f:
            pickle.dump({
                "model_objects": model_objects,
                "feature_cols":  feature_cols,
                "target_col":    target_col,
            }, f)
    except Exception as e:
        print(f"[WARN] Could not save models to /tmp: {e}")


def _load_models():
    """Load pickled models from /tmp.  Returns None if not found."""
    try:
        with open(MODELS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Static front-end
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ─────────────────────────────────────────────────────────────────────────────
# Auth — demo login
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/auth/demo", methods=["POST"])
def demo_login():
    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if username == "demo" and password == "demo123":
        session["user"] = {
            "name":     "Demo User",
            "email":    "demo@salescast.ai",
            "avatar":   None,
            "provider": "demo",
            "initial":  "D",
        }
        return jsonify({"ok": True, "user": session["user"]})

    return jsonify({"ok": False, "error": "Invalid credentials"}), 401


# ─────────────────────────────────────────────────────────────────────────────
# Auth — OAuth
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/auth/<provider>")
def oauth_start(provider):
    if provider not in OAUTH:
        return "Unknown provider", 404

    cfg = OAUTH[provider]
    if not cfg["client_id"]:
        return (
            f"""<html><body style='font-family:sans-serif;padding:40px;
                background:#0a0b0f;color:#e8eaf0'>
            <h2>⚠️ {provider.title()} OAuth Not Configured</h2>
            <p style='color:#8b92a8'>Add
            <code style='background:#181b22;padding:2px 6px;border-radius:4px'>
            {provider.upper()}_CLIENT_ID</code> and
            <code style='background:#181b22;padding:2px 6px;border-radius:4px'>
            {provider.upper()}_CLIENT_SECRET</code> to your environment variables.</p>
            <p><a href='/' style='color:#4f8ef7'>← Back to app</a></p>
            </body></html>""",
            400,
        )

    state                     = secrets.token_urlsafe(16)
    session["oauth_state"]    = state
    session["oauth_provider"] = provider

    params = {
        "client_id":     cfg["client_id"],
        "redirect_uri":  _get_redirect_uri(provider),
        "scope":         cfg["scope"],
        "state":         state,
        "response_type": "code",
    }
    if provider == "google":
        params["access_type"] = "online"
    if provider == "microsoft":
        params["response_mode"] = "query"

    return redirect(cfg["auth_url"] + "?" + urlencode(params))


@app.route("/auth/<provider>/callback")
def oauth_callback(provider):
    if provider not in OAUTH:
        return "Unknown provider", 404

    if request.args.get("state") != session.pop("oauth_state", None):
        return "Invalid OAuth state — possible CSRF attack", 400

    code = request.args.get("code")
    if not code:
        err = request.args.get("error_description",
                               request.args.get("error", "Unknown"))
        return f"OAuth error: {err}", 400

    cfg = OAUTH[provider]
    token_resp = requests.post(
        cfg["token_url"],
        data={
            "client_id":     cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code":          code,
            "redirect_uri":  _get_redirect_uri(provider),
            "grant_type":    "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    token_json   = token_resp.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return f"Failed to get access token: {token_json}", 400

    user_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept":        "application/json",
    }
    user_data = requests.get(cfg["userinfo_url"], headers=user_headers, timeout=10).json()

    if provider == "google":
        name   = user_data.get("name", "Google User")
        email  = user_data.get("email", "")
        avatar = user_data.get("picture")
    elif provider == "github":
        name   = user_data.get("name") or user_data.get("login", "GitHub User")
        email  = user_data.get("email", "")
        avatar = user_data.get("avatar_url")
        if not email:
            emails = requests.get("https://api.github.com/user/emails",
                                  headers=user_headers, timeout=10).json()
            email = next((e["email"] for e in emails if e.get("primary")), "")
    elif provider == "microsoft":
        name   = user_data.get("displayName", "MS User")
        email  = user_data.get("mail") or user_data.get("userPrincipalName", "")
        avatar = None
    else:
        name, email, avatar = "User", "", None

    session["user"] = {
        "name":     name,
        "email":    email,
        "avatar":   avatar,
        "provider": provider,
        "initial":  (name[0] if name else "?").upper(),
    }
    return redirect("/?login=success")


# ─────────────────────────────────────────────────────────────────────────────
# Auth — session info & logout
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/auth/me")
def auth_me():
    if "user" in session:
        return jsonify({"authenticated": True, "user": session["user"]})
    return jsonify({"authenticated": False})


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
# ML API — /api/train
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train_model():
    try:
        print("\n[SERVER] Received training request...")
        data = request.get_json()

        if not data:
            return _json_error("Empty request payload.")

        records      = data.get("records", [])
        target_col   = data.get("target_col")
        feature_cols = data.get("feature_cols", [])

        print(f"[INFO] Records: {len(records)}  Target: {target_col}  Features: {feature_cols}")

        df       = pd.DataFrame(records)
        required = feature_cols + [target_col]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return _json_error(f"Missing columns: {missing}")

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        print(f"[INFO] Cleaned rows: {len(df)}")

        if len(df) < 5:
            return _json_error("Insufficient numeric data after cleaning (< 5 rows).")

        result = run_pipeline(
            records=df.to_dict(orient="records"),
            feature_cols=feature_cols,
            target_col=target_col,
            models=data.get("models", ["lr", "dt", "nn"]),
            config=data.get("config", {}),
        )

        # Persist model objects to /tmp for /api/predict
        _save_models(result.get("model_objects", {}), feature_cols, target_col)

        print("[SUCCESS] Training complete.")
        response = {
            "status":       "success",
            "models":       result.get("models", {}),
            "dataset_info": result.get("dataset_info", {}),
        }
        return jsonify(convert_numpy(response))

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"Internal Server Error: {e}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# ML API — /api/predict
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict_single():
    try:
        print("\n[SERVER] Received prediction request...")
        data = request.get_json()

        if not data:
            return _json_error("Empty request payload.")

        feature_values = data.get("feature_values", [])
        feature_names  = data.get("feature_names", [])
        models_to_use  = data.get("models", [])

        if len(feature_values) != len(feature_names):
            return _json_error("Mismatch between feature values and feature names.")

        # Load models from /tmp
        stored = _load_models()
        if not stored:
            return _json_error(
                "No trained models found. Please train first — models may have "
                "expired between serverless invocations.", 404
            )

        model_objects = stored["model_objects"]
        X = np.array(feature_values, dtype=np.float32).reshape(1, -1)

        predictions = {}
        for key in (models_to_use or list(model_objects.keys())):
            if key not in model_objects:
                continue
            try:
                obj    = model_objects[key]
                scaler = obj["scaler"]
                model  = obj["model"]
                X_scaled = scaler.transform(X)
                pred     = model.predict(X_scaled)
                predictions[key] = float(pred[0])
            except Exception as e:
                print(f"[ERROR] {key}: {e}")
                predictions[key] = None

        print(f"[SUCCESS] Predictions: {predictions}")
        return jsonify({"status": "success", "predictions": predictions})

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"Prediction failed: {e}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# Health-check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "SalesCast AI"})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔮 SalesCast AI starting…  http://localhost:5000\n")
    app.run(debug=True, port=5000)
