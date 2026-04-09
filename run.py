#!/usr/bin/env python3
"""
Quick-start script for SalesCast AI.
Run:  python run.py
"""
import os
import sys

# Load .env if present
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    print("✓ Loaded .env file")
else:
    print("⚠️  No .env file found — copy .env.example to .env and fill in credentials.")
    print("   Social login will show a config error until credentials are added.")
    print("   Demo login (demo / demo123) works without any setup.\n")

# Check dependencies
try:
    import flask
    import requests
    import sklearn
    import torch
    print("✓ All dependencies found")
except ImportError:
    print("📦 Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")

from app import app

print("\n" + "="*60)
print("🔮 SalesCast AI — Machine Learning Sales Forecasting")
print("="*60)
print("   🌐 http://localhost:5000")
print("   👤 Demo login: demo / demo123")
print("   🔑 OAuth: Google, GitHub, Microsoft (optional)")
print("   🛑 Press Ctrl+C to stop")
print("="*60 + "\n")

app.run(debug=False, port=5000, host="0.0.0.0")
