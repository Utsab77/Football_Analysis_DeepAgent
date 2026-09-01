"""Baseline model 2: Random Forest classifier."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.features import FEATURE_COLUMNS

MODEL_PATH = "ml/artifacts/random_forest.joblib"


def train(train_df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model.fit(train_df[FEATURE_COLUMNS], train_df["result"])
    return model


def predict_proba(model: RandomForestClassifier, features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    proba = model.predict_proba(row)[0]
    return dict(zip(model.classes_, proba))


def save(model: RandomForestClassifier, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load(path: str = MODEL_PATH) -> RandomForestClassifier:
    return joblib.load(path)
