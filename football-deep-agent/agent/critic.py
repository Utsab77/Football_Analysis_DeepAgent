"""Critic agent: checks data completeness, contradictions, model reliability,
and unsupported conclusions before a result is returned as final.

Phase 4 (Weeks 7-8). Returns pass/fail plus reasons so the manager can
trigger re-planning on failure (see agent/manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CriticVerdict:
    passed: bool
    reasons: list[str]


def review(state) -> CriticVerdict:
    reasons = []

    # TODO (Phase 4): real checks -- e.g. did every planned step complete,
    # do the model comparison results roughly agree, is the confidence
    # calibrated, are there unsupported claims in the explanation.
    if state.result is None:
        reasons.append("No result produced")

    return CriticVerdict(passed=len(reasons) == 0, reasons=reasons)
