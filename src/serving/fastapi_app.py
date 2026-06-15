from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.serving.predictor import FraudPredictor
from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_PATH = PROJECT_ROOT / 'logs' / 'pipelines.log'
logger = get_logger(__name__, log_file=LOG_PATH)

# Request / Response schemas

class TransactionRequest(BaseModel):
    features: dict[str, float]
    threshold: float | None = None  # None = use auto-tuned threshold


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    risk_level: str
    model: str


# Lifespan — load predictor once

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading predictor...")
    app.state.predictor = FraudPredictor()
    app.state.predictor.load()
    logger.info("Predictor ready.")
    yield
    logger.info("Shutting down.")


# App

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    p = app.state.predictor
    return {"status": "ok", "model": p.run_name, "threshold": p.threshold}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    try:
        result = app.state.predictor.predict(
            request.features,
            threshold=request.threshold,
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return JSONResponse(result)
