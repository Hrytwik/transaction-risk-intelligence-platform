import math
import pandas as pd
from datetime import datetime

from db_config import get_connection
from features_phase1 import build_feature_frame, feature_cols
from models_phase1 import compute_iforest_score, predict_rf_proba


BATCH_SIZE = 10000
THRESHOLD_NORMAL = 0.5
MODEL_NAME = "rf_phase1_final"
MODEL_VERSION = "phase1_v1"   # you can change later


def fetch_raw_batch(offset: int, limit: int) -> pd.DataFrame:
    conn = get_connection()
    query = f"""
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
        ORDER BY cc_num, unix_time
        LIMIT {limit} OFFSET {offset};
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df = df.rename(columns={"lon": "long", "merch_lon": "merch_long"})
    return df



def insert_features(df_feat: pd.DataFrame):
    conn = get_connection()
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

    batch = []
    for _, row in df_feat.iterrows():
        batch.append((
            int(row["txn_id"]),
            int(row["cc_num"]),
            row["txn_datetime"] if pd.notna(row["txn_datetime"]) else None,
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
        ))

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()

    cur.close()
    conn.close()


def insert_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Insert predictions and return a DataFrame with prediction_id for each txn_id.
    """
    if df_pred.empty:
        return df_pred.assign(prediction_id=pd.Series(dtype="int64"))

    conn = get_connection()
    cur = conn.cursor()

    insert_sql = """
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

    batch = []
    for _, row in df_pred.iterrows():
        batch.append((
            int(row["txn_id"]),
            int(row["cc_num"]),
            MODEL_NAME,
            MODEL_VERSION,
            "normal",
            float(THRESHOLD_NORMAL),
            float(row["fraud_proba"]),
            int(row["predicted_label"]),
            row["risk_label"],
        ))

    cur.executemany(insert_sql, batch)
    conn.commit()

    # fetch the prediction_ids we just inserted, by txn_id
    txn_ids = tuple(df_pred["txn_id"].unique().tolist())
    # handle single element tuple SQL syntax
    if len(txn_ids) == 1:
        txn_ids_sql = f"({txn_ids[0]})"
    else:
        txn_ids_sql = str(txn_ids)

    query = f"""
        SELECT prediction_id, txn_id
        FROM model_predictions
        WHERE txn_id IN {txn_ids_sql}
          AND model_name = %s
          AND mode = %s
    """
    df_ids = pd.read_sql(query, conn, params=(MODEL_NAME, "normal"))

    cur.close()
    conn.close()

    # merge prediction_id back into df_pred
    df_pred_with_id = df_pred.merge(df_ids, on="txn_id", how="left")
    return df_pred_with_id


def insert_alerts(df_alerts: pd.DataFrame):
    """
    Insert alert records for high-risk predictions.
    Requires prediction_id to be present (NOT NULL).
    """
    if df_alerts.empty:
        return

    conn = get_connection()
    cur = conn.cursor()

    insert_sql = """
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

    batch = []
    for _, row in df_alerts.iterrows():
        batch.append((
            int(row["txn_id"]),
            int(row["prediction_id"]),      # now mandatory
            int(row["cc_num"]),
            row["risk_label"],
            float(row["fraud_proba"]),
            "normal",
        ))

    cur.executemany(insert_sql, batch)
    conn.commit()
    cur.close()
    conn.close()



def risk_from_proba(p: float, thr: float = THRESHOLD_NORMAL) -> str:
    if p < 0.3:
        return "LOW"
    elif p < thr:
        return "MEDIUM"
    else:
        return "HIGH"


def process_batch(offset: int, limit: int):
    # 1) read raw
    df_raw = fetch_raw_batch(offset, limit)
    if df_raw.empty:
        return False

    # ensure txn_datetime exists for features; derive if missing
    if "txn_datetime" not in df_raw.columns or df_raw["txn_datetime"].isna().all():
        df_raw["txn_datetime"] = pd.to_datetime(df_raw["unix_time"], unit="s")

    # 2) compute features
    df_feat = build_feature_frame(df_raw.copy())
    # keep id + meta
    df_feat["txn_id"] = df_raw["txn_id"]
    df_feat["cc_num"] = df_raw["cc_num"]
    df_feat["txn_datetime"] = df_raw["txn_datetime"]

    # 3) Isolation Forest score
    ifs = compute_iforest_score(df_feat)

    # 4) RF probabilities
    probs = predict_rf_proba(df_feat, ifs)

    df_feat["iforest_score"] = ifs

    # 5) prepare predictions DataFrame
    df_pred = pd.DataFrame({
        "txn_id": df_feat["txn_id"],
        "cc_num": df_feat["cc_num"],
        "fraud_proba": probs,
    })
    df_pred["predicted_label"] = (df_pred["fraud_proba"] >= THRESHOLD_NORMAL).astype(int)
    df_pred["risk_label"] = df_pred["fraud_proba"].apply(risk_from_proba)

    # 6) insert into DB
    insert_features(df_feat)

    # insert predictions and get prediction_id for each txn_id
    df_pred_with_id = insert_predictions(df_pred)

    # 7) create alerts for high-risk predictions, with prediction_id
    df_alerts = df_pred_with_id[df_pred_with_id["risk_label"] == "HIGH"].copy()
    insert_alerts(df_alerts)

    return True


def main():
    # count how many rows to process
    conn = get_connection()
    total = pd.read_sql("SELECT COUNT(*) AS c FROM transactions_raw;", conn)["c"].iloc[0]
    conn.close()

    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Total rows: {total}, batches of {BATCH_SIZE}: {n_batches}")

    offset = 0
    for b in range(n_batches):
        print(f"Processing batch {b+1}/{n_batches} (offset {offset}) ...")
        ok = process_batch(offset, BATCH_SIZE)
        if not ok:
            break
        offset += BATCH_SIZE

    print("Done.")


if __name__ == "__main__":
    main()
