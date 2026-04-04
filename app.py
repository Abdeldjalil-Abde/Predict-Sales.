"""
SalesCast AI - Flask Backend with OAuth 2.0
Supports: Google, GitHub, Microsoft login + demo credentials
"""

import os
import json
import secrets
import requests
from functools import wraps
from flask import (
    Flask, redirect, request, session,
    url_for, jsonify, send_from_directory
)
from urllib.parse import urlencode

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# ─── OAuth Config (loaded from env or .env file) ────────────────────────────
OAUTH = {
    "google": {
        "client_id":     os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        "auth_url":      "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url":     "https://oauth2.googleapis.com/token",
        "userinfo_url":  "https://www.googleapis.com/oauth2/v3/userinfo",
        "scope":         "openid email profile",
    },
    "github": {
        "client_id":     os.environ.get("GITHUB_CLIENT_ID", ""),
        "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
        "auth_url":      "https://github.com/login/oauth/authorize",
        "token_url":     "https://github.com/login/oauth/access_token",
        "userinfo_url":  "https://api.github.com/user",
        "scope":         "read:user user:email",
    },
    "microsoft": {
        "client_id":     os.environ.get("MICROSOFT_CLIENT_ID", ""),
        "client_secret": os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
        "auth_url":      "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url":     "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url":  "https://graph.microsoft.com/v1.0/me",
        "scope":         "openid email profile User.Read",
    },
}

# ─── Helpers ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated

def get_redirect_uri(provider):
    return url_for("oauth_callback", provider=provider, _external=True)

# ─── Static frontend ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ─── Auth: demo login ────────────────────────────────────────────────────────
@app.route("/auth/demo", methods=["POST"])
def demo_login():
    data = request.get_json()
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

# ─── Auth: OAuth start ───────────────────────────────────────────────────────
@app.route("/auth/<provider>")
def oauth_start(provider):
    if provider not in OAUTH:
        return "Unknown provider", 404
    cfg = OAUTH[provider]
    if not cfg["client_id"]:
        return f"""
        <html><body style='font-family:sans-serif;padding:40px;background:#0a0b0f;color:#e8eaf0'>
        <h2>⚠️ {provider.title()} OAuth Not Configured</h2>
        <p style='color:#8b92a8'>Add <code style='background:#181b22;padding:2px 6px;border-radius:4px'>
        {provider.upper()}_CLIENT_ID</code> and
        <code style='background:#181b22;padding:2px 6px;border-radius:4px'>
        {provider.upper()}_CLIENT_SECRET</code> to your <code>.env</code> file.</p>
        <p><a href='/' style='color:#4f8ef7'>← Back to app</a></p>
        </body></html>
        """, 400

    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    session["oauth_provider"] = provider

    params = {
        "client_id":     cfg["client_id"],
        "redirect_uri":  get_redirect_uri(provider),
        "scope":         cfg["scope"],
        "state":         state,
        "response_type": "code",
    }
    if provider == "google":
        params["access_type"] = "online"
    if provider == "microsoft":
        params["response_mode"] = "query"

    return redirect(cfg["auth_url"] + "?" + urlencode(params))

# ─── Auth: OAuth callback ────────────────────────────────────────────────────
@app.route("/auth/<provider>/callback")
def oauth_callback(provider):
    if provider not in OAUTH:
        return "Unknown provider", 404

    # Validate state
    if request.args.get("state") != session.pop("oauth_state", None):
        return "Invalid state — possible CSRF", 400

    code = request.args.get("code")
    if not code:
        error = request.args.get("error_description", request.args.get("error", "Unknown error"))
        return f"OAuth error: {error}", 400

    cfg = OAUTH[provider]

    # Exchange code for token
    token_data = {
        "client_id":     cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "code":          code,
        "redirect_uri":  get_redirect_uri(provider),
        "grant_type":    "authorization_code",
    }
    headers = {"Accept": "application/json"}
    token_resp = requests.post(cfg["token_url"], data=token_data, headers=headers, timeout=10)
    token_json = token_resp.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return f"Failed to get access token: {token_json}", 400

    # Fetch user info
    user_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    user_resp = requests.get(cfg["userinfo_url"], headers=user_headers, timeout=10)
    user_data = user_resp.json()

    # Normalize across providers
    if provider == "google":
        name   = user_data.get("name", "Google User")
        email  = user_data.get("email", "")
        avatar = user_data.get("picture")
    elif provider == "github":
        name   = user_data.get("name") or user_data.get("login", "GitHub User")
        email  = user_data.get("email", "")
        avatar = user_data.get("avatar_url")
        # GitHub may not expose email — fetch separately
        if not email:
            emails_resp = requests.get(
                "https://api.github.com/user/emails",
                headers=user_headers, timeout=10
            )
            emails = emails_resp.json()
            primary = next((e["email"] for e in emails if e.get("primary")), None)
            email = primary or ""
    elif provider == "microsoft":
        name   = user_data.get("displayName", "MS User")
        email  = user_data.get("mail") or user_data.get("userPrincipalName", "")
        avatar = None  # Graph photo requires separate call

    initial = (name[0] if name else "?").upper()

    session["user"] = {
        "name":     name,
        "email":    email,
        "avatar":   avatar,
        "provider": provider,
        "initial":  initial,
    }

    # Redirect back to frontend with success flag
    return redirect("/?login=success")

# ─── Auth: session info ───────────────────────────────────────────────────────
@app.route("/auth/me")
def auth_me():
    if "user" in session:
        return jsonify({"authenticated": True, "user": session["user"]})
    return jsonify({"authenticated": False})

# ─── Auth: logout ─────────────────────────────────────────────────────────────
@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔮 SalesCast AI starting...")
    print("   http://localhost:5000\n")
    app.run(debug=True, port=5000)
