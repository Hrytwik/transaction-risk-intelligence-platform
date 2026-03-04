# services/scoring_service.py
from datetime import datetime
from typing import Any, Dict, Tuple, List
import logging

import pandas as pd

from db_config import get_connection
from features_phase1 import build_feature_frame, feature_cols
from models_phase1 import compute_iforest_score, predict_rf_proba

# ---- logging config ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("fraud_scoring")

# ---- constants aligned with batch_feature_and_score.py ----
BATCH_THRESHOLD_NORMAL = 0.5
MODEL_NAME = "rf_phase1_final"
MODEL_VERSION = "phase1_v1"


def risk_from_proba(p: float, thr: float = BATCH_THRESHOLD_NORMAL) -> str:
    if p < 0.3:
        return "LOW"
    elif p < thr:
        return "MEDIUM"
    else:
        return "HIGH"


def _insert_transaction_raw(txn_dict: Dict[str, Any]) -> int:
    """
    Insert one row into transactions_raw and return txn_id.
    Mirrors load_csv_to_mysql, but for single-row, API path.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # derive txn_datetime if not sent
        if txn_dict.get("txn_datetime") is None:
            txn_dict["txn_datetime"] = datetime.utcfromtimestamp(txn_dict["unix_time"])

        insert_sql = """
            INSERT INTO transactions_raw (
                cc_num, unix_time, txn_datetime, amt,
                merchant_id, merchant_name, category,
                lat, lon, merch_lat, merch_lon,
                is_fraud_label, source_system
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        params = (
            int(txn_dict["cc_num"]),
            int(txn_dict["unix_time"]),
            txn_dict["txn_datetime"],
            float(txn_dict["amt"]),
            txn_dict.get("merchant_id"),
            txn_dict.get("merchant_name"),
            txn_dict.get("category"),
            float(txn_dict["lat"]),
            float(txn_dict["lon"]),
            float(txn_dict["merch_lat"]),
            float(txn_dict["merch_lon"]),
            None,  # online path: no label
            txn_dict.get("source_system") or "api_v1",
        )

        cur.execute(insert_sql, params)
        conn.commit()
        txn_id = cur.lastrowid
        cur.close()

        logger.info(
            "Inserted txn_id=%s for cc_num=%s source_system=%s",
            txn_id,
            txn_dict.get("cc_num"),
            txn_dict.get("source_system"),
        )

        return txn_id
    finally:
        conn.close()


def _fetch_recent_history(cc_num: int, limit: int = 20) -> pd.DataFrame:
    """
    Fetch recent history for this card, including the new txn.
    """
    conn = get_connection()
    try:
        query = """
            SELECT
                txn_id,
                cc_num,
                unix_time,
                txn_datetime,
                amt,
                merchant_name,
                category,
                lat,
                lon,
                merch_lat,
                merch_lon,
                is_fraud_label
            FROM transactions_raw
            WHERE cc_num = %s
            ORDER BY unix_time DESC
            LIMIT %s;
        """
        df = pd.read_sql(query, conn, params=(cc_num, limit))
        # reverse so that time is ascending for feature functions
        df = df.iloc[::-1].reset_index(drop=True)
        # rename lon/merch_lon to long/merch_long for features_phase1
        df = df.rename(columns={"lon": "long", "merch_lon": "merch_long"})
        return df
    finally:
        conn.close()


def _insert_features(df_feat: pd.DataFrame):
    """
    Same logic as insert_features in batch_feature_and_score.py, but reused here.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        insert_sql = """
            INSERT INTO transactions_features (
                txn_id,
                cc_num,
                txn_datetime,
                amt,
                time_since_prev,
                amt_roll_mean_10,
                amt_roll_std_10,
                txn_count_10,
                dist_cust_merchant_km,
                amt_day_sum,
                amt_day_count,
                amt_day_max,
                iforest_score
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        batch: List[tuple] = []
        for _, row in df_feat.iterrows():
            batch.append(
                (
                    int(row["txn_id"]),
                    int(row["cc_num"]),
                    row["txn_datetime"],
                    float(row["amt"]),
                    float(row["time_since_prev"]),
                    float(row["amt_roll_mean_10"]),
                    float(row["amt_roll_std_10"]),
                    int(row["txn_count_10"]),
                    float(row["dist_cust_merchant_km"]),
                    float(row["amt_day_sum"]),
                    int(row["amt_day_count"]),
                    float(row["amt_day_max"]),
                    float(row["iforest_score"]),
                )
            )

        if batch:
            cur.executemany(insert_sql, batch)
            conn.commit()

        cur.close()
        logger.info(
            "Inserted features for %d rows (txn_id=%s...)", len(batch), df_feat["txn_id"].iloc[0]
        )
    finally:
        conn.close()


def _insert_predictions_and_alerts(df_pred: pd.DataFrame) -> Tuple[int, bool, int | None]:
    """
    Insert one prediction row and optional alert row, return (prediction_id, alert_created, alert_id).
    Assumes df_pred has columns: txn_id, cc_num, fraud_proba, predicted_label, risk_label.
    """
    assert len(df_pred) == 1
    row = df_pred.iloc[0]

    conn = get_connection()
    try:
        cur = conn.cursor()

        # Insert prediction
        insert_pred_sql = """
            INSERT INTO model_predictions (
                txn_id,
                cc_num,
                model_name,
                model_version,
                mode,
                threshold,
                fraud_proba,
                predicted_label,
                risk_label
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        pred_params = (
            int(row["txn_id"]),
            int(row["cc_num"]),
            MODEL_NAME,
            MODEL_VERSION,
            "normal",
            float(BATCH_THRESHOLD_NORMAL),
            float(row["fraud_proba"]),
            int(row["predicted_label"]),
            row["risk_label"],
        )
        cur.execute(insert_pred_sql, pred_params)
        conn.commit()
        prediction_id = cur.lastrowid

        alert_created = False
        alert_id = None

        # Optional alert
        if row["risk_label"] == "HIGH":
            insert_alert_sql = """
                INSERT INTO alerts (
                    txn_id,
                    prediction_id,
                    cc_num,
                    risk_label,
                    fraud_proba,
                    mode
                )
                VALUES (%s,%s,%s,%s,%s,%s)
            """
            alert_params = (
                int(row["txn_id"]),
                int(prediction_id),
                int(row["cc_num"]),
                row["risk_label"],
                float(row["fraud_proba"]),
                "normal",
            )
            cur.execute(insert_alert_sql, alert_params)
            conn.commit()
            alert_id = cur.lastrowid
            alert_created = True

        cur.close()

        logger.info(
            "Stored prediction_id=%s for txn_id=%s (alert_created=%s, alert_id=%s)",
            prediction_id,
            row["txn_id"],
            alert_created,
            alert_id,
        )

        return prediction_id, alert_created, alert_id
    finally:
        conn.close()


def score_raw_df_for_online(txn_dict: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Main entry point for the API.
    1) Insert into transactions_raw -> txn_id
    2) Fetch recent history for that cc_num
    3) Build features
    4) Score with IF + RF
    5) Insert into transactions_features, model_predictions, alerts
    6) Return txn_id and result dict
    """
    # 1) Insert raw transaction
    txn_id = _insert_transaction_raw(txn_dict)

    # 2) Fetch history (including this txn)
    cc_num = int(txn_dict["cc_num"])
    df_raw = _fetch_recent_history(cc_num=cc_num, limit=20)

    if df_raw.empty:
        logger.error("No history fetched for cc_num=%s after insert", cc_num)
        raise RuntimeError("No history fetched for cc_num after insert")

    # 3) Features
    if "txn_datetime" not in df_raw.columns or df_raw["txn_datetime"].isna().all():
        df_raw["txn_datetime"] = pd.to_datetime(df_raw["unix_time"], unit="s")

    df_feat_full = build_feature_frame(df_raw.copy())
    df_feat_full["txn_id"] = df_raw["txn_id"]
    df_feat_full["cc_num"] = df_raw["cc_num"]
    df_feat_full["txn_datetime"] = df_raw["txn_datetime"]

    df_feat = df_feat_full[df_feat_full["txn_id"] == txn_id].copy()
    if df_feat.empty:
        df_feat = df_feat_full.tail(1).copy()
        logger.warning(
            "Could not find row for txn_id=%s after feature building; using last row fallback",
            txn_id,
        )

    # 4) model scoring
    ifs = compute_iforest_score(df_feat)
    probs = predict_rf_proba(df_feat, ifs)
    df_feat["iforest_score"] = ifs

    fraud_proba = float(probs[0])
    predicted_label = int(fraud_proba >= BATCH_THRESHOLD_NORMAL)
    risk_label = risk_from_proba(fraud_proba)

    logger.info(
        "Scored txn_id=%s cc_num=%s fraud_proba=%.4f predicted_label=%s risk_label=%s",
        txn_id,
        df_feat["cc_num"].iloc[0],
        fraud_proba,
        predicted_label,
        risk_label,
    )

    # 5) write features
    _insert_features(df_feat)

    # predictions + alerts
    df_pred = pd.DataFrame(
        {
            "txn_id": [int(df_feat["txn_id"].iloc[0])],
            "cc_num": [int(df_feat["cc_num"].iloc[0])],
            "fraud_proba": [fraud_proba],
            "predicted_label": [predicted_label],
            "risk_label": [risk_label],
        }
    )
    prediction_id, alert_created, alert_id = _insert_predictions_and_alerts(df_pred)

    # 6) build result
    result = {
        "txn_id": txn_id,
        "fraud_proba": fraud_proba,
        "predicted_label": predicted_label,
        "risk_label": risk_label,
        "mode": "normal",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": BATCH_THRESHOLD_NORMAL,
        "alert_created": alert_created,
        "alert_id": alert_id,
    }
    return txn_id, result
