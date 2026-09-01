"""Evaluation metrics for the prediction models.

Keep this separate from agent evaluation (see evaluation/evaluator.py) --
a good prediction model does not imply a good agent, and vice versa.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def probabilistic_metrics(y_true_encoded, y_proba, classes: list[str]) -> dict:
    """y_true_encoded: true labels as class indices matching `classes` order.
    y_proba: array of shape (n_samples, n_classes).
    """
    return {
        "log_loss": log_loss(y_true_encoded, y_proba, labels=list(range(len(classes)))),
    }


def brier_score_per_class(y_true: pd.Series, y_proba_class: pd.Series, positive_class: str) -> float:
    binary_true = (y_true == positive_class).astype(int)
    return brier_score_loss(binary_true, y_proba_class)
