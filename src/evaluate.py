"""
src/evaluate.py
─────────────────────────────────────────────────────────────
Evaluation helpers: metrics, plots, reports.
Used by notebooks and the training script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)


# ─────────────────────────────────────────────────────────────
# Full metric dict
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "f1"       : f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall"   : recall_score(y_true, y_pred),
    }
    if y_proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
        m["avg_precision"] = average_precision_score(y_true, y_proba)
    return m


def print_report(y_true, y_pred, model_name: str = "Model"):
    print(f"\n{'='*55}")
    print(f"  Evaluation Report – {model_name}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model",
                           save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred Fake", "Pred Real"],
        yticklabels=["True Fake", "True Real"],
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_roc_curves(models_data: list, save_path: str | None = None):
    """
    models_data: list of (name, y_true, y_proba) tuples
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for i, (name, y_true, y_proba) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})", color=colors[i % len(colors)], lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_model_comparison(results: dict, save_path: str | None = None):
    """
    results: {model_name: metrics_dict}
    """
    metric_names = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    model_names  = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, metrics) in enumerate(results.items()):
        vals = [metrics.get(m, 0) for m in metric_names]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)], edgecolor="black")

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – All Metrics", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig
