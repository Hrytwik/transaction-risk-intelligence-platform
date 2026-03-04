# app/schemas/scoring.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, condecimal


class TransactionIn(BaseModel):
    """
    Shape of one incoming transaction, similar to transactions_raw.
    """
    cc_num: int
    unix_time: int = Field(..., description="Unix timestamp in seconds")
    amt: condecimal(gt=0) = Field(..., description="Transaction amount")
    merchant_id: Optional[int] = None
    merchant_name: Optional[str] = None
    category: Optional[str] = None
    lat: float
    lon: float
    merch_lat: float
    merch_lon: float
    txn_datetime: Optional[datetime] = None
    source_system: Optional[str] = "api_v1"


class ScoreTransactionResponse(BaseModel):
    txn_id: int
    fraud_proba: float
    predicted_label: int
    risk_label: str
    mode: str
    model_name: str
    model_version: str
    threshold: float
    alert_created: bool
    alert_id: Optional[int] = None
