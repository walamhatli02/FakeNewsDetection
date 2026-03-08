"""
src/predict.py
─────────────────────────────────────────────────────────────
Inference utilities – used by the FastAPI endpoint and the
test scripts.  Loads model artifacts once and exposes a
clean predict() function.
"""

import os
import json
import pickle
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.preprocess import clean_text, extract_features_single

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str             # "REAL" | "FAKE"
    label_id: int          # 1 = Real, 0 = Fake
    confidence: float      # probability of the predicted class
    real_probability: float
    fake_probability: float


class FakeNewsPredictor:
    """
    Wrapper that loads artifacts once and exposes a simple predict() API.

    Usage
    -----
    predictor = FakeNewsPredictor(data_dir="data/")
    result = predictor.predict("Headline text", "Article body text...")
    print(result.label, result.confidence)
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._model         = None
        self._tfidf         = None
        self._feature_cols  = None
        self._subject_enc   = None
        self._load_artifacts()

    # ── Artifact loading ──────────────────────────────────────

    def _load_artifacts(self):
        model_path  = os.path.join(self.data_dir, "best_model.pkl")
        tfidf_path  = os.path.join(self.data_dir, "tfidf_vectorizer.pkl")
        fcols_path  = os.path.join(self.data_dir, "feature_cols.json")
        senc_path   = os.path.join(self.data_dir, "subject_encoder.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run the training pipeline first:\n"
                "  python -m src.train"
            )

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            self._tfidf = pickle.load(f)
        with open(fcols_path) as f:
            self._feature_cols = json.load(f)

        if os.path.exists(senc_path):
            with open(senc_path, "rb") as f:
                self._subject_enc = pickle.load(f)

        log.info(f"✅  Predictor loaded – {len(self._feature_cols)} feature columns")

    # ── Core prediction ───────────────────────────────────────

    def predict(
        self,
        title: str,
        text: str,
        subject: Optional[str] = None,
    ) -> Prediction:
        """
        Predict whether a single article is Fake or Real.

        Parameters
        ----------
        title   : Headline of the article
        text    : Full body text
        subject : Optional topic category

        Returns
        -------
        Prediction dataclass
        """
        import re
        text = re.sub(r'^[A-Z\s]+\([^)]+\)\s*-\s*', '', str(text))
        combined = clean_text(title) + " " + clean_text(text)
        

        X_tfidf = self._tfidf.transform([combined])
        meta    = extract_features_single(
            title, text, self._feature_cols, subject, self._subject_enc
        )
        X_meta  = csr_matrix(np.array(meta, dtype=np.float32).reshape(1, -1))
        X       = hstack([X_tfidf, X_meta])

        label_id = int(self._model.predict(X)[0])
        proba    = self._model.predict_proba(X)[0]

        return Prediction(
            label            = "REAL" if label_id == 1 else "FAKE",
            label_id         = label_id,
            confidence       = float(proba[label_id]),
            real_probability = float(proba[1]),
            fake_probability = float(proba[0]),
        )

    def predict_batch(
        self,
        articles: List[dict],
    ) -> List[Prediction]:
        """
        Predict a list of dicts with keys: title, text, (subject).
        """
        return [
            self.predict(
                a.get("title", ""),
                a.get("text", ""),
                a.get("subject"),
            )
            for a in articles
        ]
