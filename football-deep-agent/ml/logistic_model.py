"""Baseline model 1: Logistic Regression (multinomial, for H/D/A)."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.features import FEATURE_COLUMNS

MODEL_PATH = "ml/artifacts/logistic_model.joblib"


def train(train_df: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(train_df[FEATURE_COLUMNS], train_df["result"])
    return model


def predict_proba(model: LogisticRegression, features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    proba = model.predict_proba(row)[0]
    return dict(zip(model.classes_, proba))


def save(model: LogisticRegression, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load(path: str = MODEL_PATH) -> LogisticRegression:
    return joblib.load(path)
