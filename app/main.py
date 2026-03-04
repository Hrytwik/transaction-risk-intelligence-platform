from fastapi import FastAPI
from app.api.routes_scoring import router as scoring_router

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.include_router(scoring_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Fraud detection service"}
