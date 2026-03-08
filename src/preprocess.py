"""
src/preprocess.py
─────────────────────────────────────────────────────────────
Reusable preprocessing pipeline for the Fake News Detection project.
Used by both the training notebooks and the FastAPI inference endpoint.
"""

import re
import string
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise raw text: lowercase, remove URLs / HTML / numbers / punctuation."""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags
    text = re.sub(r"\d+", " NUM ", text)                    # Numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Punct
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# Handcrafted features
# ─────────────────────────────────────────────────────────────

def _uppercase_ratio(text: str) -> float:
    text = str(text)
    return sum(1 for c in text if c.isupper()) / max(len(text), 1)


def _avg_word_len(text: str) -> float:
    words = text.split()
    return float(np.mean([len(w) for w in words])) if words else 0.0


def _unique_word_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / max(len(words), 1)


FEATURE_COLS = [
    "num_exclamations",
    "num_questions",
    "uppercase_ratio",
    "avg_word_len",
    "text_word_count",
    "title_word_count",
    "unique_word_ratio",
]


def extract_handcrafted_features(df: pd.DataFrame, include_subject: bool = True) -> pd.DataFrame:
    """
    Add handcrafted numeric feature columns to *df* (in-place).
    Returns the modified dataframe.
    """
    df["clean_title"]   = df["title"].apply(clean_text)
    df["clean_text"]    = df["text"].apply(clean_text)
    df["combined"]      = df["clean_title"] + " " + df["clean_text"]

    df["num_exclamations"] = df["text"].apply(lambda x: str(x).count("!"))
    df["num_questions"]    = df["text"].apply(lambda x: str(x).count("?"))
    df["uppercase_ratio"]  = df["text"].apply(_uppercase_ratio)
    df["avg_word_len"]     = df["clean_text"].apply(_avg_word_len)
    df["text_word_count"]  = df["clean_text"].apply(lambda x: len(x.split()))
    df["title_word_count"] = df["clean_title"].apply(lambda x: len(x.split()))
    df["unique_word_ratio"]= df["clean_text"].apply(_unique_word_ratio)

    return df


def extract_features_single(
    title: str,
    text: str,
    feature_cols: List[str],
    subject: Optional[str] = None,
    subject_encoder: Optional[LabelEncoder] = None,
) -> List[float]:
    """
    Extract the same handcrafted features for a single article at inference time.
    Returns a list aligned with *feature_cols*.
    """
    c_text  = clean_text(text)
    c_title = clean_text(title)

    feat_map = {
        "num_exclamations" : text.count("!"),
        "num_questions"    : text.count("?"),
        "uppercase_ratio"  : _uppercase_ratio(text),
        "avg_word_len"     : _avg_word_len(c_text),
        "text_word_count"  : len(c_text.split()),
        "title_word_count" : len(c_title.split()),
        "unique_word_ratio": _unique_word_ratio(c_text),
    }

    if "subject_enc" in feature_cols and subject_encoder is not None:
        try:
            feat_map["subject_enc"] = int(
                subject_encoder.transform([subject or "unknown"])[0]
            )
        except Exception:
            feat_map["subject_enc"] = 0

    return [feat_map.get(col, 0.0) for col in feature_cols]


# ─────────────────────────────────────────────────────────────
# Full pipeline: load → engineer → vectorise → split
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    tfidf: Optional[TfidfVectorizer] = None,
    subject_encoder: Optional[LabelEncoder] = None,
    fit: bool = True,
    include_subject: bool = True,
    tfidf_kwargs: Optional[dict] = None,
) -> Tuple[sp.csr_matrix, TfidfVectorizer, List[str], Optional[LabelEncoder]]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df          : DataFrame with columns [title, text, (subject)]
    tfidf       : Existing TF-IDF vectorizer (set fit=False to use it for transform only)
    fit         : If True, fit the TF-IDF on df; otherwise transform only
    tfidf_kwargs: Override TF-IDF constructor arguments

    Returns
    -------
    X           : Sparse feature matrix (TF-IDF + handcrafted)
    tfidf       : Fitted (or passed-through) TF-IDF vectorizer
    feature_cols: List of handcrafted feature column names
    subject_enc : Fitted LabelEncoder (or None)
    """
    df = df.copy()
    df = extract_handcrafted_features(df, include_subject=include_subject)

    local_feature_cols = list(FEATURE_COLS)

    # Subject encoding
    if include_subject and "subject" in df.columns:
        if fit:
            subject_encoder = LabelEncoder()
            df["subject_enc"] = subject_encoder.fit_transform(df["subject"].fillna("unknown"))
        else:
            df["subject_enc"] = df["subject"].apply(
                lambda x: int(subject_encoder.transform([x or "unknown"])[0])
                if subject_encoder else 0
            )
        local_feature_cols.append("subject_enc")

    X_meta = csr_matrix(df[local_feature_cols].values.astype(np.float32))

    # TF-IDF
    default_tfidf_kwargs = dict(
        max_features  = 50_000,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 5,
        max_df        = 0.9,
        strip_accents = "unicode",
    )
    if tfidf_kwargs:
        default_tfidf_kwargs.update(tfidf_kwargs)

    if fit:
        tfidf = TfidfVectorizer(**default_tfidf_kwargs)
        X_tfidf = tfidf.fit_transform(df["combined"])
    else:
        X_tfidf = tfidf.transform(df["combined"])

    X = hstack([X_tfidf, X_meta]).tocsr()
    return X, tfidf, local_feature_cols, subject_encoder

def load_and_prepare(data_dir: str, test_size: float = 0.2, random_state: int = 42,
                     tfidf_kwargs: Optional[dict] = None, include_subject: bool = False):
    """
    Convenience function: load True/Fake CSVs, merge, engineer features,
    vectorise and split into train/test.
    """
    import os

    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    df = (
        pd.concat([true_df, fake_df], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    # Remove Reuters dateline to avoid data leakage
    df["text"] = df["text"].str.replace(r'^[A-Z\s]+\([^)]+\)\s*-\s*', '', regex=True)

    y = df["label"]

    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

    X_train, tfidf, feature_cols, subject_enc = build_feature_matrix(
    train_df, fit=True, tfidf_kwargs=tfidf_kwargs, include_subject=include_subject
)
    X_test,  _,     _,            _           = build_feature_matrix(
    test_df, tfidf=tfidf, subject_encoder=subject_enc, fit=False, include_subject=include_subject
)

    y_train = train_df["label"]
    y_test  = test_df["label"]

    return X_train, X_test, y_train, y_test, tfidf, feature_cols, subject_enc, df
