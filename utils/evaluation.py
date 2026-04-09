"""
utils/evaluation.py

Central evaluation utilities for all models.

All models should use this to ensure:
- consistent metrics
- fair comparison
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.

    Returns
    -------
    dict
        accuracy, precision, recall, f1
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",  # important for class balance
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def print_metrics(metrics):
    """
    Nicely print metrics.
    """
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")