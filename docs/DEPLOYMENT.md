# Deployment notes

**Current status: the dashboard runs locally. It is NOT deployed anywhere.** Do not claim a live deployment until
one exists and the URL has been opened and checked.

## 1. Local run (the supported way)

```powershell
cd C:\Users\sushm\federated-learning-educational-analytics
.\.venv\Scripts\Activate.ps1
python main.py all          # only needed once, or after code changes
streamlit run dashboard/app.py
```

Streamlit prints a local URL (normally <http://localhost:8501>). Use `--server.port 8502` if 8501 is busy.

## 2. Streamlit Community Cloud (free, recommended if you deploy)

Requirements: the repository must be public on GitHub (or you grant the Streamlit app access), and every file the
app needs must be committed.

What the app needs at runtime:

| Need | Committed? | Notes |
|---|---|---|
| `dashboard/`, `src/` | yes | code |
| `results/models/{full,early}/*` | yes | preprocessor, schema, seed-42 models (tiny) |
| `results/reports/*` incl. `runs/` | yes | all metrics and histories |
| `results/figures/*` | yes | only used by the README |
| `data/raw/student-mat.csv` | yes | CC BY 4.0 (checked on the UCI page, Nov 2026); attribution is in the README. The Dataset Analytics page reads it. |
| `data/processed/*` | no | not needed by the dashboard |

Steps:

1. Push `main` to GitHub (repository must be public for friends to open the app without a Streamlit login).
2. Go to <https://share.streamlit.io>, sign in with the **SushmithaB-06** GitHub account, "Create app" → "Deploy a
   public app from GitHub".
3. Repository `SushmithaB-06/federated-learning-educational-analytics`, branch `main`, main file path
   `dashboard/app.py`. Choose an app URL (e.g. `fl-educational-analytics`).
4. Advanced settings → Python 3.11 (3.12 also works). Leave secrets empty (the app has none).
5. Deploy. The first build takes several minutes; `requirements.txt` already points at CPU-only PyTorch wheels so
   the install stays small.
6. Open the URL, click through all eight pages, press Predict, and only then write the URL into the README.

Limitations of the free tier: 1 GB RAM, app sleeps after inactivity, cold start of 30-60 s, no GPU (not needed).

## 3. Other options

- **Hugging Face Spaces (Streamlit SDK)**: same requirements, similar free tier; put the app entry in `app.py` at the
  repo root or set `app_file` in the Space's README.
- **Docker** (any VPS):

  ```Dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0"]
  ```

## 4. Secrets and configuration

The application uses no API keys, databases or credentials. `.streamlit/config.toml` only sets the theme. Never
commit `.streamlit/secrets.toml` (already ignored).

## 5. Reproducing the results from a clean machine

```powershell
git clone https://github.com/SushmithaB-06/federated-learning-educational-analytics.git
cd federated-learning-educational-analytics
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# place student-mat.csv in data\raw\
python main.py all
python -m pytest
```

The pipeline is deterministic on CPU: a re-run of seed 42 reproduces the saved JSON results and model weights exactly
(verified on Windows 11, Python 3.11, torch 2.15). Different PyTorch/CPU combinations may produce tiny floating-point
differences; the mean ± std tables are robust to those.
