import pandas as pd
from db_config import get_connection

csv_path = "archive (1)/fraudTrain.csv"   # update this

df = pd.read_csv(csv_path)

# create datetime from unix_time
df["txn_datetime"] = pd.to_datetime(df["unix_time"], unit="s")

cols = [
    "cc_num", "unix_time", "txn_datetime", "amt",
    "merchant", "category",
    "lat", "long", "merch_lat", "merch_long", "is_fraud"
]

df = df[cols].rename(columns={
    "long": "lon",
    "merchant": "merchant_name",
    "is_fraud": "is_fraud_label"
})

insert_sql = """
    INSERT INTO transactions_raw (
        cc_num, unix_time, txn_datetime, amt,
        merchant_id, merchant_name, category,
        lat, lon, merch_lat, merch_lon, is_fraud_label, source_system
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

conn = get_connection()
cur = conn.cursor()

batch = []
for _, row in df.iterrows():
    batch.append((
        int(row["cc_num"]),
        int(row["unix_time"]),
        row["txn_datetime"].to_pydatetime(),
        float(row["amt"]),
        None,                         # no merchant_id column in CSV
        row["merchant_name"],
        row["category"],
        float(row["lat"]),
        float(row["lon"]),
        float(row["merch_lat"]),
        float(row["merch_long"]),
        int(row["is_fraud_label"]),
        "csv_phase1"
    ))

    if len(batch) >= 1000:
        cur.executemany(insert_sql, batch)
        conn.commit()
        batch = []

if batch:
    cur.executemany(insert_sql, batch)
    conn.commit()

cur.close()
conn.close()
