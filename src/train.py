"""
src/train.py
─────────────────────────────────────────────────────────────
Standalone training script. Run from the project root:

    python -m src.train --data_dir data/ --output_dir data/ --experiment fake_news

This script mirrors Notebook 3 but is importable and usable from the CLI.
"""

import argparse
import os
import pickle
import json
import logging

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_and_prepare

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "f1"       : f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall"   : recall_score(y_true, y_pred),
    }
    if y_proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
    return m


# ─────────────────────────────────────────────────────────────
# Individual model trainers
# ─────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_test, y_test):
    params = {"C": 1.0, "max_iter": 1000, "solver": "saga", "random_state": 42, "n_jobs": -1}
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_params(params)
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_test, model.predict(X_test),
                                   model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        log.info(f"  LogisticRegression → accuracy={metrics['accuracy']:.4f}  auc={metrics.get('roc_auc',0):.4f}")
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "logloss", "random_state": 42, "tree_method": "hist",
    }
    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        metrics = compute_metrics(y_test, model.predict(X_test),
                                   model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        log.info(f"  XGBoost            → accuracy={metrics['accuracy']:.4f}  auc={metrics.get('roc_auc',0):.4f}")
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": 500, "max_depth": 7, "learning_rate": 0.05,
        "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }
    with mlflow.start_run(run_name="LightGBM") as run:
        mlflow.log_params(params)
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )
        metrics = compute_metrics(y_test, model.predict(X_test),
                                   model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model")
        log.info(f"  LightGBM           → accuracy={metrics['accuracy']:.4f}  auc={metrics.get('roc_auc',0):.4f}")
        run_id = run.info.run_id
    return model, metrics, run_id


# ─────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────

def run_training(
    data_dir: str = "data",
    output_dir: str = "data",
    experiment_name: str = "fake_news_detection",
    mlflow_uri: str = "sqlite:///mlflow.db",
):
    log.info("=" * 60)
    log.info("  FAKE NEWS DETECTION – TRAINING PIPELINE")
    log.info("=" * 60)

    # 1. Load & preprocess
    log.info("Step 1/4 – Loading and preprocessing data …")
    (
        X_train, X_test, y_train, y_test,
        tfidf, feature_cols, subject_enc, df
    ) = load_and_prepare(data_dir)

    log.info(f"  Train : {X_train.shape[0]:,} samples  |  Test : {X_test.shape[0]:,} samples")
    log.info(f"  Features: {X_train.shape[1]:,}")

    # 2. MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # 3. Train all models
    log.info("Step 2/4 – Training models …")
    lr_model,   lr_metrics            = train_logistic_regression(X_train, y_train, X_test, y_test)
    xgb_model,  xgb_metrics           = train_xgboost(X_train, y_train, X_test, y_test)
    lgbm_model, lgbm_metrics, run_id  = train_lightgbm(X_train, y_train, X_test, y_test)

    # 4. Pick best model (by ROC-AUC)
    log.info("Step 3/4 – Selecting best model …")
    candidates = [
        ("LogisticRegression", lr_model,   lr_metrics),
        ("XGBoost",            xgb_model,  xgb_metrics),
        ("LightGBM",           lgbm_model, lgbm_metrics),
    ]
    best_name, best_model, best_metrics = max(candidates, key=lambda t: t[2].get("roc_auc", 0))
    log.info(f"  Best model: {best_name}  (ROC-AUC={best_metrics.get('roc_auc',0):.4f})")

    # 5. Save artifacts
    log.info("Step 4/4 – Saving artifacts …")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.pkl")
    tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    fcols_path = os.path.join(output_dir, "feature_cols.json")
    meta_path  = os.path.join(output_dir, "training_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    with open(fcols_path, "w") as f:
        json.dump(feature_cols, f)
    if subject_enc is not None:
        with open(os.path.join(output_dir, "subject_encoder.pkl"), "wb") as f:
            pickle.dump(subject_enc, f)

    meta = {
        "best_model"  : best_name,
        "metrics"     : best_metrics,
        "n_train"     : int(X_train.shape[0]),
        "n_test"      : int(X_test.shape[0]),
        "n_features"  : int(X_train.shape[1]),
        "feature_cols": feature_cols,
        "mlflow_run_id": run_id,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"  ✅ best_model.pkl         → {model_path}")
    log.info(f"  ✅ tfidf_vectorizer.pkl   → {tfidf_path}")
    log.info(f"  ✅ feature_cols.json      → {fcols_path}")
    log.info(f"  ✅ training_meta.json     → {meta_path}")
    log.info("=" * 60)
    log.info("  Training complete!")
    log.info(f"  Accuracy : {best_metrics['accuracy']:.4f}")
    log.info(f"  F1       : {best_metrics['f1']:.4f}")
    log.info(f"  ROC-AUC  : {best_metrics.get('roc_auc',0):.4f}")
    log.info("=" * 60)

    return best_model, best_metrics


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection model")
    parser.add_argument("--data_dir",    default="data",               help="Folder with True.csv / Fake.csv")
    parser.add_argument("--output_dir",  default="data",               help="Folder to save model artifacts")
    parser.add_argument("--experiment",  default="fake_news_detection", help="MLflow experiment name")
    parser.add_argument("--mlflow_uri",  default="sqlite:///mlflow.db", help="MLflow tracking URI")
    args = parser.parse_args()

    run_training(
        data_dir       = args.data_dir,
        output_dir     = args.output_dir,
        experiment_name= args.experiment,
        mlflow_uri     = args.mlflow_uri,
    )
