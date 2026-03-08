"""
tests/test_preprocess.py
─────────────────────────────────────────────────────────────
Unit tests for the preprocessing module.
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from src.preprocess import (
    clean_text,
    extract_handcrafted_features,
    extract_features_single,
    FEATURE_COLS,
)


# ─────────────────────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Visit https://example.com for details")
        assert "http" not in result
        assert "example" not in result

    def test_removes_html(self):
        result = clean_text("Text with <b>bold</b> and <br/> tags")
        assert "<" not in result and ">" not in result

    def test_replaces_numbers(self):
        result = clean_text("There are 42 items in 2023")
        assert "42" not in result
        assert "2023" not in result
        assert "NUM" in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! This is... a test?")
        assert "," not in result and "!" not in result and "?" not in result

    def test_handles_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_handles_nan(self):
        assert clean_text(float("nan")) == ""

    def test_strips_whitespace(self):
        result = clean_text("  lots   of   spaces  ")
        assert result == "lots of spaces"


# ─────────────────────────────────────────────────────────────
# extract_handcrafted_features
# ─────────────────────────────────────────────────────────────

class TestHandcraftedFeatures:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "title": ["Federal Reserve raises rates", "SHOCKING!!! Deep State EXPOSED!!!"],
            "text":  [
                "The Federal Reserve raised rates by 0.25 percent on Wednesday.",
                "WAKE UP SHEEPLE!!! They are HIDING the truth!!! SHARE THIS NOW!!!",
            ],
            "subject": ["politics", "news"],
            "label": [1, 0],
        })

    def test_returns_dataframe(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_creates_all_feature_cols(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_creates_combined_column(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        assert "combined" in result.columns

    def test_exclamation_count(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        assert result["num_exclamations"].iloc[0] == 0   # real news
        assert result["num_exclamations"].iloc[1] > 3    # fake news

    def test_uppercase_ratio_fake_higher(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        assert result["uppercase_ratio"].iloc[1] > result["uppercase_ratio"].iloc[0]

    def test_all_values_finite(self, sample_df):
        result = extract_handcrafted_features(sample_df)
        for col in FEATURE_COLS:
            assert np.all(np.isfinite(result[col].values)), f"Non-finite in {col}"


# ─────────────────────────────────────────────────────────────
# extract_features_single (inference)
# ─────────────────────────────────────────────────────────────

class TestExtractFeaturesSingle:

    def test_returns_list_of_correct_length(self):
        result = extract_features_single("Test title", "Some text here.", FEATURE_COLS)
        assert isinstance(result, list)
        assert len(result) == len(FEATURE_COLS)

    def test_all_finite(self):
        result = extract_features_single("Breaking news!!!", "SHOCKING TRUTH!!!??!", FEATURE_COLS)
        assert all(np.isfinite(v) for v in result)

    def test_consistent_with_batch_features(self):
        title = "Government announces new policy"
        text  = "The government today announced a significant policy change."

        single_result = extract_features_single(title, text, FEATURE_COLS)

        df = pd.DataFrame([{"title": title, "text": text, "label": 1}])
        batch_result = extract_handcrafted_features(df)

        for i, col in enumerate(FEATURE_COLS):
            assert abs(single_result[i] - batch_result[col].iloc[0]) < 1e-6, \
                f"Mismatch for {col}: single={single_result[i]}, batch={batch_result[col].iloc[0]}"
