import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns or "value" not in df.columns:
        raise ValueError("Need columns: time, value")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)

def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    df = _ensure_sorted(df)
    X = df[["value"]].values
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    labels = clf.fit_predict(X)  # -1 anomaly, 1 normal
    scores = -clf.decision_function(X)  # higher = more anomalous
    out = df.copy()
    out["is_anomaly"] = labels == -1
    out["score"] = scores
    return out

def detect_zscore(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    df = _ensure_sorted(df)
    mu = df["value"].mean()
    sigma = df["value"].std(ddof=1) or 1e-8
    z = (df["value"] - mu) / sigma
    out = df.copy()
    out["is_anomaly"] = z.abs() >= z_thresh
    out["score"] = z.abs()
    return out

def detect_iqr(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    df = _ensure_sorted(df)
    q1 = df["value"].quantile(0.25)
    q3 = df["value"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = df.copy()
    out["is_anomaly"] = (df["value"] < lo) | (df["value"] > hi)
    # score = distance outside the band (0 if inside)
    dist = np.where(df["value"] < lo, lo - df["value"],
                    np.where(df["value"] > hi, df["value"] - hi, 0.0))
    out["score"] = dist
    return out
