#!/usr/bin/env python3
"""
Quick-start script for SalesCast AI.
Run:  python run.py
"""
import os, sys

# Load .env if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    print("✅ Loaded .env")
else:
    print("⚠️  No .env file found — copy .env.example to .env and fill in credentials.")
    print("   Social login will show a config error until credentials are added.")
    print("   Demo login (demo / demo123) works without any setup.\n")

# Check dependencies
try:
    import flask, requests
except ImportError:
    print("📦 Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")

from app import app
print("🔮 SalesCast AI — http://localhost:5000")
print("   Demo login: demo / demo123")
print("   Press Ctrl+C to stop\n")
app.run(debug=False, port=5000)
