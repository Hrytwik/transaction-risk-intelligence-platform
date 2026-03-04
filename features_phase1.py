import numpy as np
import pandas as pd

# Final numeric feature list used in Phase 1
feature_cols = [
    "amt",
    "time_since_prev",
    "amt_roll_mean_10",
    "amt_roll_std_10",
    "txn_count_10",
    "dist_cust_merchant_km",
    "amt_day_sum",
    "amt_day_count",
    "amt_day_max",
]


def add_time_since_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time_since_prev per cc_num using unix_time.
    Assumes df has columns: cc_num, unix_time.
    """
    df = df.sort_values(["cc_num", "unix_time"]).copy()
    df["time_since_prev"] = (
        df.groupby("cc_num")["unix_time"].diff().fillna(0)
    )
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Haversine distance in km between two sets of coordinates.
    Inputs are numpy arrays or pandas Series in degrees.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_spatial_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dist_cust_merchant_km from lat, lon, merch_lat, merch_long.
    """
    df = df.copy()
    df["dist_cust_merchant_km"] = _haversine_km(
        df["lat"].astype(float),
        df["long"].astype(float),
        df["merch_lat"].astype(float),
        df["merch_long"].astype(float),
    )
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Rolling stats over last `window` transactions per card on amt.
    Assumes df is already sorted by cc_num, unix_time.
    """
    df = df.copy()
    grp = df.groupby("cc_num")["amt"]

    df["amt_roll_mean_10"] = grp.transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df["amt_roll_std_10"] = grp.transform(
        lambda x: x.rolling(window, min_periods=1).std().fillna(0)
    )
    df["txn_count_10"] = grp.transform(
        lambda x: x.rolling(window, min_periods=1).count()
    )
    return df


def add_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily behavior per card: sum, count, max of amt per (cc_num, day).
    """
    df = df.copy()

    # derive datetime from unix_time if needed
    if "txn_datetime" in df.columns:
        dt = pd.to_datetime(df["txn_datetime"])
    else:
        dt = pd.to_datetime(df["unix_time"], unit="s")

    df["txn_date"] = dt.dt.date

    agg = (
        df.groupby(["cc_num", "txn_date"])["amt"]
        .agg(
            amt_day_sum="sum",
            amt_day_count="count",
            amt_day_max="max",
        )
        .reset_index()
    )

    # merge back the aggregates
    df = df.merge(
        agg,
        on=["cc_num", "txn_date"],
        how="left",
    )

    return df



def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col in ["amt", "lat", "long", "merch_lat", "merch_long"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df = add_time_since_prev(df)
    df = add_rolling_features(df)
    df = add_spatial_feature(df)
    df = add_daily_aggregates(df)

    return df

