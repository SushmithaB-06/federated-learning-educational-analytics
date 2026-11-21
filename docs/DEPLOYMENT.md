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
| `data/raw/student-mat.csv` | **no** | the Dataset Analytics page reads it. Either (a) commit it if the UCI licence on the dataset page permits redistribution, or (b) add a download step. See below. |
| `data/processed/*` | no | not needed by the dashboard |

Steps:

1. Push `main` to GitHub.
2. Decide how the raw CSV reaches the app (option a or b above). For option (a) remove `data/raw/*` from
   `.gitignore`, commit the file, and cite the licence in the README.
3. Go to <https://share.streamlit.io>, sign in with GitHub, "New app", choose the repo, branch `main`,
   main file `dashboard/app.py`.
4. Advanced settings → Python version 3.11. Leave secrets empty (the app has none).
5. Deploy. First build takes several minutes because of PyTorch. If it fails on memory, pin a CPU-only torch build in
   `requirements.txt`, e.g. add a line `--extra-index-url https://download.pytorch.org/whl/cpu` above `torch`.
6. Open the URL, click through all eight pages, and only then write the URL into the README.

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
