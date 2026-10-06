"""FastAPI service for Amazon review sentiment predictions."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sentiment-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
model: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model once when the application starts."""
    global model

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception:
        model = None
        logger.exception("Could not load model from %s", MODEL_PATH)

    yield


app = FastAPI(
    title="Amazon Review Sentiment API",
    description="Predict positive or negative sentiment from an Amazon review.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    review: str = Field(
        min_length=1,
        max_length=10_000,
        examples=["This product is excellent and works perfectly."],
    )


class PredictionResponse(BaseModel):
    prediction: int
    sentiment: str
    confidence: float


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Amazon Review Sentiment API",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    loaded = model is not None
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, int | str | float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    started_at = time.perf_counter()
    prediction = int(model.predict([request.review])[0])
    probabilities = model.predict_proba([request.review])[0]
    confidence = float(max(probabilities))
    sentiment = "positive" if prediction == 1 else "negative"
    duration_ms = (time.perf_counter() - started_at) * 1_000

    logger.info(
        "prediction=%s sentiment=%s confidence=%.4f duration_ms=%.2f",
        prediction,
        sentiment,
        confidence,
        duration_ms,
    )

    return {
        "prediction": prediction,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
    }
