# Federated Learning for Educational Analytics

Final-year B.Tech project. A **simulated** federated learning system in which three
simulated educational institutions train a student-performance model locally on the
UCI Student Performance dataset, and a central server combines the models with
Federated Averaging (FedAvg). Raw student records never leave the client.

> Status: rebuild in progress. Full documentation is written in Phase 10.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# place student-mat.csv in data\raw\  (download link in src/config.py)
python main.py verify
```
