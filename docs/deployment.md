# Deployment Guide

This document describes how Workforce360 is deployed to Streamlit Community Cloud.

## Platform

- **Hosting**: Streamlit Community Cloud
- **Source Control**: GitHub
- **CI/CD**: GitHub Actions (for linting and tests); Streamlit auto-deploys on push to `main`.

## Prerequisites

- A GitHub account.
- A Streamlit Community Cloud account (sign in with GitHub).
- This repository pushed to GitHub.

## Secrets Management

Sensitive configuration (API keys, database URLs, etc.) is stored in `.streamlit/secrets.toml`, which is **not** committed to Git (ignored via `.gitignore`).

Example `secrets.toml` (local):

```toml
[llm]
api_key = "sk-..."
openai_org = "org-..."

[database]
url = "postgresql://user:pass@host:5432/workforce360"
```

On Streamlit Community Cloud:

1. Go to your app’s settings in the Streamlit dashboard.
2. Open **Advanced Settings** → **Secrets**.
3. Paste the contents of your local `secrets.toml` there.
4. Save.

The app accesses secrets via `st.secrets` in code:

```python
import streamlit as st

llm_api_key = st.secrets["llm"]["api_key"]
db_url = st.secrets["database"]["url"]
```

## Deploying to Streamlit Community Cloud

1. Ensure your app entrypoint is `app/app.py` and `requirements.txt` is up to date.
2. Push all changes to the `main` branch on GitHub.
3. In the Streamlit dashboard:
   - Click **Create app**.
   - Select this repository.
   - Set:
     - Branch: `main`
     - File path: `app/app.py`
   - Configure Python version if needed.
   - Add secrets as described above.
4. Click **Deploy**.

Streamlit will build and launch your app. The URL will be something like:

```text
[https://workforce360.streamlit.app](https://workforce360.streamlit.app)
```

## Continuous Deployment

- Every `git push` to `main` triggers an automatic rebuild and redeployment on Streamlit Community Cloud.
- GitHub Actions runs linting and tests on each push/PR (see `.github/workflows/ci.yml`).

## Local Testing Before Deploy

Before pushing:

```bash
# Activate venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app/app.py
```

Verify functionality locally, then push to `main` to deploy.

## Troubleshooting

- **Import errors**: Ensure all required packages are in `requirements.txt`.
- **Secrets not found**: Confirm `secrets.toml` is correctly configured in Streamlit’s Advanced Settings.
- **Port issues**: Locally, Streamlit runs on port 8501 by default; in production, Streamlit manages this.