"""
api/main.py  –  FastAPI backend for Fake News Detection (Week 4)
Run: uvicorn api.main:app --reload --port 8000
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from src.predict import FakeNewsPredictor, Prediction

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="🔍 Fake News Detection API",
    description="Detect whether a news article is REAL or FAKE using LightGBM.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

predictor: Optional[FakeNewsPredictor] = None

@app.on_event("startup")
def startup_event():
    global predictor
    data_dir = os.environ.get("DATA_DIR", "data")
    try:
        predictor = FakeNewsPredictor(data_dir=data_dir)
        log.info("✅ Model loaded.")
    except FileNotFoundError as e:
        log.warning(f"⚠️  Model not found – run training first. {e}")

# ── Schemas ────────────────────────────────────────────────────────────────
class NewsInput(BaseModel):
    title:   str = Field(..., min_length=5, max_length=500,
                         example="Federal Reserve raises interest rates")
    text:    str = Field(..., min_length=20, max_length=50_000,
                         example="The Federal Reserve raised rates on Wednesday...")
    subject: Optional[str] = Field(None, example="politics")

class BatchInput(BaseModel):
    articles: List[NewsInput] = Field(..., min_items=1, max_items=50)

class PredictionOut(BaseModel):
    label: str;  label_id: int;  confidence: float
    real_probability: float;  fake_probability: float

def _check():
    if predictor is None:
        raise HTTPException(503, "Model not loaded. Run: python -m src.train")

def _out(p: Prediction) -> PredictionOut:
    return PredictionOut(label=p.label, label_id=p.label_id, confidence=p.confidence,
                         real_probability=p.real_probability, fake_probability=p.fake_probability)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {"message": "🔍 Fake News Detection API is running", "docs": "/docs", "version": "1.0.0"}

@app.get("/health", tags=["Info"])
def health():
    return {"status": "healthy", "model": "LightGBM", "version": "1.0.0", "ready": predictor is not None}

@app.get("/examples", tags=["Info"])
def examples():
    return {
        "real": {"title": "Federal Reserve raises interest rates by 0.25 percent",
                 "text": "The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday. Fed Chair Jerome Powell said the decision was unanimous among voting members. The central bank said it remains strongly committed to returning inflation to its 2 percent target.",
                 "subject": "politics"},
        "fake": {"title": "SHOCKING!!! Government puts MICROCHIPS in vaccines to control YOUR MIND!!!",
                 "text": "WAKE UP SHEEPLE!!! The deep state globalists have been secretly injecting microchips into vaccines since 2020!!! A WHISTLEBLOWER revealed the TRUTH that mainstream media is HIDING from you!! Share this before it gets DELETED!!! Bill Gates admitted the chips will be activated by 5G!!!",
                 "subject": "health"},
    }

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(news: NewsInput):
    """Predict whether a single article is REAL or FAKE."""
    _check()
    return _out(predictor.predict(news.title, news.text, news.subject))

@app.post("/predict/batch", response_model=List[PredictionOut], tags=["Prediction"])
def predict_batch(body: BatchInput):
    """Predict up to 50 articles at once."""
    _check()
    articles = [{"title": a.title, "text": a.text, "subject": a.subject} for a in body.articles]
    return [_out(p) for p in predictor.predict_batch(articles)]

@app.post("/explain", tags=["Prediction"])
def explain(news: NewsInput):
    """Return LIME explanation for a single article."""
    _check()
    from lime.lime_text import LimeTextExplainer
    from scipy.sparse import hstack, csr_matrix
    from src.preprocess import clean_text, extract_features_single
    import numpy as np

    combined = clean_text(news.title) + " " + clean_text(news.text)

    def predict_fn(texts):
        results = []
        for t in texts:
            X_tfidf = predictor._tfidf.transform([t])
            meta = extract_features_single(news.title, news.text, predictor._feature_cols, news.subject, predictor._subject_enc)
            X_meta = csr_matrix(np.array(meta, dtype=np.float32).reshape(1, -1))
            X = hstack([X_tfidf, X_meta])
            results.append(predictor._model.predict_proba(X)[0])
        return np.array(results)

    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    exp = explainer.explain_instance(combined, predict_fn, num_features=8, num_samples=300)
    features = [{"word": f[0], "weight": round(f[1], 4)} for f in exp.as_list()]
    return {"explanation": features}
