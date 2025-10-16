import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from detectors import detect_isolation_forest, detect_zscore, detect_iqr

st.set_page_config(page_title="Anomaly Monitor", layout="wide")
st.title("🔎 Anomaly Monitor")

with st.sidebar:
    st.header("1) Data")
    data_src = st.radio("Load data from", ["Upload CSV", "Generate demo"])
    if data_src == "Upload CSV":
        up = st.file_uploader("CSV with columns: time, value", type=["csv"])
    else:
        up = None

    st.header("2) Detector")
    model = st.selectbox("Method", ["IsolationForest", "Z-score", "IQR"])
    contamination = st.slider("Contamination (IF only)", 0.001, 0.20, 0.02, 0.001)
    z_thresh = st.slider("Z-score threshold", 2.0, 6.0, 3.0, 0.1)
    iqr_mult = st.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)

    st.header("3) Options")
    smooth = st.checkbox("Add rolling mean (window=5)", value=True)
    downsample = st.checkbox("Downsample preview (keep every 3rd row)", value=False)

# --- Load data ---
if up is not None:
    df = pd.read_csv(up)
else:
    # generate a synthetic demo time series
    n = 600
    rng = pd.date_range("2024-01-01", periods=n, freq="H")
    base = np.sin(np.linspace(0, 18*np.pi, n)) * 10 + 50
    noise = np.random.normal(0, 2, n)
    vals = base + noise
    # inject anomalies
    spikes_idx = np.random.choice(np.arange(n), size=12, replace=False)
    vals[spikes_idx] += np.random.normal(15, 6, size=len(spikes_idx))
    df = pd.DataFrame({"time": rng, "value": vals})

# Validate columns
if not {"time", "value"}.issubset(df.columns):
    st.error("CSV must contain 'time' and 'value' columns.")
    st.stop()

# Ensure proper dtypes
df = df.copy()
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

if smooth:
    df["value_smooth"] = df["value"].rolling(5, min_periods=1).mean()
    feature_col = "value_smooth"
else:
    feature_col = "value"

# --- Detect anomalies ---
if model == "IsolationForest":
    res = detect_isolation_forest(df[["time", feature_col]].rename(columns={feature_col: "value"}),
                                  contamination=contamination)
elif model == "Z-score":
    res = detect_zscore(df[["time", feature_col]].rename(columns={feature_col: "value"}),
                        z_thresh=z_thresh)
else:
    res = detect_iqr(df[["time", feature_col]].rename(columns={feature_col: "value"}),
                     k=iqr_mult)

# Defensive checks: ensure detector returned expected DataFrame columns
if not isinstance(res, pd.DataFrame):
    st.error("Detector must return a pandas DataFrame with columns: time, is_anomaly, score")
    st.stop()
missing = {"time", "is_anomaly", "score"} - set(res.columns)
if missing:
    st.error(f"Detector result missing columns: {', '.join(sorted(missing))}")
    st.stop()

# ensure time is datetime so merge behaves
res = res.copy()
res["time"] = pd.to_datetime(res["time"])

# Merge back to full df
df = df.merge(res[["time", "is_anomaly", "score"]], on="time", how="left")
df["is_anomaly"] = df["is_anomaly"].fillna(False)

# Preview dataframe (create after merging so it contains is_anomaly)
df_preview = df.copy()
if downsample:
    df_preview = df_preview.iloc[::3, :]

# --- Visualization ---
st.subheader("Interactive chart")
fig = px.line(df_preview, x="time", y=feature_col, title="Signal")
anoms = df_preview[df_preview["is_anomaly"]]
if not anoms.empty:
    fig.add_scatter(x=anoms["time"], y=anoms[feature_col], mode="markers",
                    marker=dict(size=10, symbol="x"),
                    name="Anomaly")
st.plotly_chart(fig, width='stretch')

# --- Summary ---
st.subheader("Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Anomalies", int(df["is_anomaly"].sum()))
with col3:
    pct = 100 * df["is_anomaly"].mean()
    st.metric("Anomaly %", f"{pct:.2f}%")

st.dataframe(df.tail(20), width='stretch')

# --- Download annotated CSV ---
st.subheader("Export")
out = df[["time", "value", "is_anomaly", "score"]]
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Download annotated CSV", csv, file_name="anomalies.csv", mime="text/csv")

# --- (Optional) API call placeholder ---
with st.expander("🔌 Send results to API (demo placeholder)"):
    st.write("In a real setup, POST `out` to your FastAPI endpoint for storage/reporting.")
