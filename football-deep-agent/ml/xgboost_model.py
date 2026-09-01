"""Baseline model 3: XGBoost classifier.

Requires `xgboost` (see requirements.txt). XGBoost needs numeric labels,
so we encode H/D/A -> 0/1/2 and decode on the way out.
"""
from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBClassifier

from ml.features import FEATURE_COLUMNS

MODEL_PATH = "ml/artifacts/xgboost_model.joblib"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def train(train_df: pd.DataFrame) -> XGBClassifier:
    y = train_df["result"].map(LABEL_MAP)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )
    model.fit(train_df[FEATURE_COLUMNS], y)
    return model


def predict_proba(model: XGBClassifier, features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    proba = model.predict_proba(row)[0]
    return {INV_LABEL_MAP[i]: p for i, p in enumerate(proba)}


def save(model: XGBClassifier, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load(path: str = MODEL_PATH) -> XGBClassifier:
    return joblib.load(path)
