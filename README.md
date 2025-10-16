# Anomaly Monitor

This small Streamlit app helps you detect and explore anomalies in a single time series.

## What it does
- Load a CSV with columns `time` and `value` (or use the demo generator).
- Choose a detection method: Isolation Forest, Z-score, or IQR.
- Optionally smooth the series with a rolling mean.
- View interactive charts and download an annotated CSV with `is_anomaly` and `score`.

## Sample CSV
`sample_data.csv` contains hourly values with a couple of injected spikes you can upload from the app sidebar.

## Why this is useful
- Quick anomaly exploration: helps spot unusual events in logs, metrics, or sensor data.
- Baseline for monitoring: lets you compare simple statistical methods quickly before deploying more complex models.
- Lightweight: runs locally with Streamlit and minimal dependencies.

## How to run
```powershell
pip install -r requirements.txt
streamlit run .\app.py
```
## Create a GitHub repository and push
If you want to publish this project to GitHub, run these commands from the project root in PowerShell (replace <your-repo-url> with the new remote URL):

```powershell
git init
git add .
git commit -m "Initial import: anomaly detection side project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

You can name the repository `anomaly-detection-side-project` or pick another name you prefer.

---
License: MIT
