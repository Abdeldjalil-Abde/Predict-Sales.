# 🔮 SalesCast AI — Setup Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in credentials
cp .env.example .env

# 3. Run
python run.py
# → open http://localhost:5000
```

Demo login works immediately with **demo / demo123** — no setup needed.

---

## Setting Up Real OAuth (Google, GitHub, Microsoft)

### 🔴 Google

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or select existing)
3. **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
4. Application type: **Web application**
5. Add Authorized redirect URI:
   ```
   http://localhost:5000/auth/google/callback
   ```
6. Copy **Client ID** and **Client Secret** into `.env`:
   ```
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-secret
   ```
7. Also enable the **Google+ API** or **People API** in APIs & Services → Library

---

### ⚫ GitHub

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. **OAuth Apps → New OAuth App**
3. Fill in:
   - Application name: `SalesCast AI`
   - Homepage URL: `http://localhost:5000`
   - Authorization callback URL: `http://localhost:5000/auth/github/callback`
4. Copy **Client ID** and generate a **Client Secret**, paste into `.env`:
   ```
   GITHUB_CLIENT_ID=your-client-id
   GITHUB_CLIENT_SECRET=your-secret
   ```

---

### 🔵 Microsoft

1. Go to [Azure Portal](https://portal.azure.com/)
2. **Azure Active Directory → App registrations → New registration**
3. Name: `SalesCast AI` | Account type: **Any Azure AD directory + personal Microsoft accounts**
4. Redirect URI: **Web** → `http://localhost:5000/auth/microsoft/callback`
5. After creation go to **Certificates & secrets → New client secret**
6. Paste into `.env`:
   ```
   MICROSOFT_CLIENT_ID=your-application-id
   MICROSOFT_CLIENT_SECRET=your-secret-value
   ```

---

## Deploying to Production

When deploying (Heroku, Railway, Render, VPS), update the redirect URIs in each provider's
console to match your domain, e.g. `https://yourdomain.com/auth/google/callback`.

Set environment variables on your host instead of using `.env`.

## Project Structure

```
salescast/
├── app.py              ← Flask backend + OAuth routes
├── run.py              ← Easy start script
├── requirements.txt
├── .env.example        ← Copy to .env and fill in credentials
└── static/
    └── index.html      ← Full frontend (SalesCast AI dashboard)
```
