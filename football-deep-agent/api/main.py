"""FastAPI app. Added in Phase 4+, once the agent works via CLI first.

Run: uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Football Deep Agent API")


class AnalyzeRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    # TODO: wire this to agent.manager.run() once the agent loop exists.
    from ml.predict import predict_match

    return predict_match(req.home_team, req.away_team)
