"""
tests/test_api.py
─────────────────────────────────────────────────────────────
Integration tests for the FastAPI endpoints.

These tests use TestClient and mock the predictor so they run
WITHOUT needing the trained model files.

Run: pytest tests/test_api.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch the predictor before importing the app
from src.predict import Prediction

MOCK_REAL = Prediction(
    label="REAL", label_id=1, confidence=0.98,
    real_probability=0.98, fake_probability=0.02
)
MOCK_FAKE = Prediction(
    label="FAKE", label_id=0, confidence=0.96,
    real_probability=0.04, fake_probability=0.96
)


@pytest.fixture(scope="module")
def client():
    with patch("api.main.FakeNewsPredictor") as MockPredictor:
        instance = MockPredictor.return_value
        instance.predict.return_value = MOCK_REAL
        instance.predict_batch.side_effect = lambda articles: [MOCK_REAL if i % 2 == 0 else MOCK_FAKE for i, _ in enumerate(articles)]

        from api.main import app, predictor
        import api.main as api_module
        api_module.predictor = instance

        with TestClient(app) as c:
            yield c


# ─────────────────────────────────────────────────────────────
# Health / Root
# ─────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status"  in data
        assert "model"   in data
        assert "ready"   in data
        assert "version" in data

    def test_examples_endpoint(self, client):
        resp = client.get("/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert "real" in data
        assert "fake" in data


# ─────────────────────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    REAL_PAYLOAD = {
        "title"  : "Federal Reserve raises interest rates by 0.25 percent",
        "text"   : ("The Federal Reserve raised its benchmark interest rate on Wednesday. "
                    "Fed Chair Jerome Powell said the central bank remains committed to "
                    "returning inflation to its 2 percent target."),
        "subject": "politics",
    }

    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=self.REAL_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json=self.REAL_PAYLOAD)
        data = resp.json()
        assert "label"            in data
        assert "label_id"         in data
        assert "confidence"       in data
        assert "real_probability" in data
        assert "fake_probability" in data

    def test_label_is_real_or_fake(self, client):
        resp = client.post("/predict", json=self.REAL_PAYLOAD)
        assert resp.json()["label"] in {"REAL", "FAKE"}

    def test_probabilities_sum_to_one(self, client):
        resp = client.post("/predict", json=self.REAL_PAYLOAD)
        data = resp.json()
        total = data["real_probability"] + data["fake_probability"]
        assert abs(total - 1.0) < 0.01

    def test_confidence_between_0_and_1(self, client):
        resp = client.post("/predict", json=self.REAL_PAYLOAD)
        conf = resp.json()["confidence"]
        assert 0 <= conf <= 1

    def test_missing_title_returns_422(self, client):
        resp = client.post("/predict", json={"text": "Some text here"})
        assert resp.status_code == 422

    def test_missing_text_returns_422(self, client):
        resp = client.post("/predict", json={"title": "Some title"})
        assert resp.status_code == 422

    def test_title_too_short_returns_422(self, client):
        resp = client.post("/predict", json={"title": "Hi", "text": "Some long text content here"})
        assert resp.status_code == 422

    def test_text_too_short_returns_422(self, client):
        resp = client.post("/predict", json={"title": "Valid title here", "text": "Short"})
        assert resp.status_code == 422

    def test_optional_subject(self, client):
        payload = {k: v for k, v in self.REAL_PAYLOAD.items() if k != "subject"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────
# /predict/batch
# ─────────────────────────────────────────────────────────────

class TestPredictBatchEndpoint:
    def test_batch_returns_200(self, client):
        payload = {"articles": [
            {"title": "Real news headline here", "text": "The government announced today that it would increase funding for public schools across the country."},
            {"title": "SHOCKING FAKE NEWS!!!", "text": "WAKE UP SHEEPLE!!! They are hiding the truth from you!!! Share this before it gets deleted!!!"},
        ]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_returns_list(self, client):
        payload = {"articles": [
            {"title": "Real news headline here", "text": "The government announced today that it would increase funding for public schools across the country."},
        ]}
        resp = client.post("/predict/batch", json=payload)
        assert isinstance(resp.json(), list)

    def test_batch_same_length_as_input(self, client):
        articles = [
            {"title": f"Article {i} title here", "text": f"This is article number {i} with enough text content for processing."}
            for i in range(5)
        ]
        payload = {"articles": articles}
        resp = client.post("/predict/batch", json=payload)
        assert len(resp.json()) == 5