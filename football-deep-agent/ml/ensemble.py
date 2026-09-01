"""Combine baseline model predictions (simple probability averaging to start)."""
from __future__ import annotations


def average_ensemble(*proba_dicts: dict) -> dict:
    """Average class probabilities across any number of {class: proba} dicts."""
    if not proba_dicts:
        raise ValueError("Need at least one probability dict")
    classes = proba_dicts[0].keys()
    return {
        cls: sum(d[cls] for d in proba_dicts) / len(proba_dicts)
        for cls in classes
    }
