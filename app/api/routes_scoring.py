# app/api/routes_scoring.py
from fastapi import APIRouter, HTTPException
import logging

from app.schemas.scoring import TransactionIn, ScoreTransactionResponse
from services.scoring_service import score_raw_df_for_online

router = APIRouter()
logger = logging.getLogger("fraud_api")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/score_transaction", response_model=ScoreTransactionResponse)
async def score_transaction(payload: TransactionIn):
    try:
        logger.info("Received score request for cc_num=%s", payload.cc_num)
        txn_dict = payload.model_dump()
        _, result = score_raw_df_for_online(txn_dict)
        logger.info(
            "Completed score for cc_num=%s txn_id=%s fraud_proba=%.4f risk_label=%s",
            payload.cc_num,
            result["txn_id"],
            result["fraud_proba"],
            result["risk_label"],
        )
        return ScoreTransactionResponse(**result)
    except Exception as e:
        logger.exception("Error scoring transaction for cc_num=%s", payload.cc_num)
        raise HTTPException(status_code=500, detail=str(e))
