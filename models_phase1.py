import joblib
import numpy as np
import pandas as pd

from features_phase1 import feature_cols


# ---- load saved Phase‑1 artifacts once ----
# Adjust paths if they are in a different directory
rf = joblib.load("rf_phase1_final.joblib")
iso = joblib.load("iforest_phase1_final.joblib")
scaler_if = joblib.load("iforest_scaler_phase1.joblib")


def compute_iforest_score(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Compute Isolation Forest anomaly scores for new data using
    the Phase‑1 scaler and IsolationForest model.

    Expects df_feat to contain exactly feature_cols.
    Returns a 1D numpy array: higher = more anomalous
    (same convention as Phase‑1: -decision_function).
    """
    X_if = df_feat[feature_cols].astype("float32")
    X_scaled = scaler_if.transform(X_if)
    scores = iso.decision_function(X_scaled)   # higher = more normal
    iforest_score = -scores                    # higher = more anomalous
    return iforest_score


def predict_rf_proba(df_feat: pd.DataFrame, iforest_score: np.ndarray) -> np.ndarray:
    """
    Compute fraud probability with the saved RandomForest using
    feature_cols + iforest_score in the same order as Phase‑1.
    """
    sup_features = feature_cols + ["iforest_score"]

    df_sup = df_feat.copy()
    df_sup["iforest_score"] = iforest_score

    X = df_sup[sup_features].astype("float32")
    probs = rf.predict_proba(X)[:, 1]
    return probs
