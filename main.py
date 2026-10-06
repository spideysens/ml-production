"""FastAPI service for Amazon review sentiment predictions."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Serve a small browser-based demo for the prediction API."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Review Sentiment Analyzer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(14, 28, 48, 0.88);
      --line: rgba(148, 163, 184, 0.2);
      --text: #f8fafc;
      --muted: #9fb0c5;
      --accent: #7c3aed;
      --accent-2: #22d3ee;
      --positive: #34d399;
      --negative: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 28px 18px;
      background:
        radial-gradient(circle at 15% 15%, rgba(34,211,238,.16), transparent 30%),
        radial-gradient(circle at 85% 80%, rgba(124,58,237,.2), transparent 32%),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(760px, 100%); }
    .eyebrow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
      color: var(--muted);
      font-size: 14px;
    }
    .brand { color: var(--text); font-weight: 750; letter-spacing: .02em; }
    .status { display: inline-flex; align-items: center; gap: 8px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #fbbf24; box-shadow: 0 0 14px currentColor; }
    .dot.online { background: var(--positive); }
    .card {
      padding: clamp(24px, 5vw, 48px);
      border: 1px solid var(--line);
      border-radius: 26px;
      background: var(--panel);
      box-shadow: 0 28px 80px rgba(0,0,0,.35);
      backdrop-filter: blur(18px);
    }
    h1 { margin: 0; font-size: clamp(34px, 7vw, 62px); line-height: .98; letter-spacing: -.05em; }
    h1 span { color: var(--accent-2); }
    .intro { margin: 18px 0 30px; color: var(--muted); line-height: 1.7; max-width: 620px; }
    label { display: block; margin-bottom: 10px; font-weight: 700; }
    textarea {
      width: 100%; min-height: 150px; resize: vertical;
      padding: 17px 18px;
      border: 1px solid var(--line); border-radius: 15px;
      background: rgba(2, 8, 23, .62); color: var(--text);
      font: inherit; font-size: 16px; line-height: 1.55; outline: none;
      transition: border-color .2s, box-shadow .2s;
    }
    textarea:focus { border-color: var(--accent-2); box-shadow: 0 0 0 4px rgba(34,211,238,.1); }
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 22px; }
    .sample {
      border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px;
      background: transparent; color: var(--muted); cursor: pointer;
    }
    .sample:hover { color: var(--text); border-color: rgba(34,211,238,.55); }
    #analyze {
      width: 100%; border: 0; border-radius: 14px; padding: 15px 20px;
      background: linear-gradient(110deg, var(--accent), #2563eb);
      color: white; font: inherit; font-weight: 800; cursor: pointer;
      box-shadow: 0 12px 32px rgba(124,58,237,.28);
    }
    #analyze:hover { filter: brightness(1.08); transform: translateY(-1px); }
    #analyze:disabled { cursor: wait; opacity: .65; transform: none; }
    .result {
      display: none; margin-top: 22px; padding: 20px;
      border: 1px solid var(--line); border-radius: 16px;
      background: rgba(2,8,23,.48);
    }
    .result.show { display: block; animation: rise .25s ease-out; }
    .result-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    .sentiment { margin: 0; font-size: 25px; text-transform: capitalize; }
    .confidence { color: var(--muted); font-variant-numeric: tabular-nums; }
    .meter { height: 8px; margin-top: 15px; overflow: hidden; border-radius: 99px; background: rgba(148,163,184,.14); }
    .meter > div { width: 0; height: 100%; border-radius: inherit; transition: width .5s ease; }
    .result.positive .sentiment { color: var(--positive); }
    .result.positive .meter > div { background: var(--positive); }
    .result.negative .sentiment { color: var(--negative); }
    .result.negative .meter > div { background: var(--negative); }
    .error { color: var(--negative); }
    footer { display: flex; justify-content: center; gap: 20px; margin-top: 20px; font-size: 14px; }
    footer a { color: var(--muted); text-decoration: none; }
    footer a:hover { color: var(--accent-2); }
    @keyframes rise { from { opacity: 0; transform: translateY(8px); } }
    @media (max-width: 520px) { .result-head { align-items: flex-start; flex-direction: column; gap: 5px; } }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">
      <span class="brand">AMAZON REVIEW AI</span>
      <span class="status"><span id="status-dot" class="dot"></span><span id="status-text">Checking model…</span></span>
    </div>
    <section class="card">
      <h1>What does your review <span>feel like?</span></h1>
      <p class="intro">Enter an Amazon product review and the trained machine-learning model will classify its sentiment and show its confidence.</p>
      <form id="form">
        <label for="review">Product review</label>
        <textarea id="review" maxlength="10000" required placeholder="Example: The product arrived quickly and works perfectly."></textarea>
        <div class="samples" aria-label="Example reviews">
          <button class="sample" type="button" data-review="This product is excellent and works perfectly. I would definitely recommend it.">Try a positive review</button>
          <button class="sample" type="button" data-review="Very disappointing quality. It stopped working after two days and I want a refund.">Try a negative review</button>
        </div>
        <button id="analyze" type="submit">Analyze sentiment</button>
      </form>
      <div id="result" class="result" role="status" aria-live="polite">
        <div class="result-head">
          <h2 id="sentiment" class="sentiment"></h2>
          <span id="confidence" class="confidence"></span>
        </div>
        <div class="meter"><div id="meter-fill"></div></div>
      </div>
    </section>
    <footer><a href="/docs">Interactive API docs</a><a href="/health">Health check</a></footer>
  </main>
  <script>
    const form = document.querySelector('#form');
    const review = document.querySelector('#review');
    const button = document.querySelector('#analyze');
    const result = document.querySelector('#result');
    const sentiment = document.querySelector('#sentiment');
    const confidence = document.querySelector('#confidence');
    const meter = document.querySelector('#meter-fill');

    document.querySelectorAll('.sample').forEach((sample) => {
      sample.addEventListener('click', () => { review.value = sample.dataset.review; review.focus(); });
    });

    fetch('/health').then((response) => response.json()).then((data) => {
      if (data.model_loaded) {
        document.querySelector('#status-dot').classList.add('online');
        document.querySelector('#status-text').textContent = 'Model online';
      } else {
        document.querySelector('#status-text').textContent = 'Model unavailable';
      }
    }).catch(() => { document.querySelector('#status-text').textContent = 'Service unavailable'; });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = review.value.trim();
      if (!text) return;
      button.disabled = true;
      button.textContent = 'Analyzing…';
      result.className = 'result';
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ review: text })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Prediction failed');
        const percent = (data.confidence * 100).toFixed(1);
        sentiment.className = 'sentiment';
        sentiment.textContent = data.sentiment + ' sentiment';
        confidence.textContent = percent + '% confidence';
        result.className = 'result show ' + data.sentiment;
        requestAnimationFrame(() => { meter.style.width = percent + '%'; });
      } catch (error) {
        sentiment.textContent = 'Could not analyze review';
        sentiment.className = 'sentiment error';
        confidence.textContent = error.message;
        meter.style.width = '0';
        result.className = 'result show';
      } finally {
        button.disabled = false;
        button.textContent = 'Analyze sentiment';
      }
    });
  </script>
</body>
</html>"""


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
