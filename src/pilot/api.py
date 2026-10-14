"""FastAPI HTTP service for PeakWhale™ Pilot."""
from __future__ import annotations

from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .engine import ForecastResult, forecast_csv, forecast_dataframe, list_examples, load_example

app = FastAPI(
    title="PeakWhale™ Pilot API",
    description="Domain-agnostic outcome forecaster and early-warning engine with plain-English driver attributions.",
    version="1.0.0",
)


class ForecastRequest(BaseModel):
    rows: list[dict[str, Any]] | None = None
    csv_text: str | None = None
    example: str | None = None
    target_column: str | None = None
    id_column: str | None = None
    feature_columns: list[str] | None = None
    risk_threshold: float | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "peakwhale-pilot"}


@app.get("/api/examples")
def get_examples() -> dict[str, list[str]]:
    return {"examples": list_examples()}


@app.post("/api/forecast", response_model=ForecastResult)
def post_forecast(req: ForecastRequest) -> ForecastResult:
    try:
        if req.example:
            df = load_example(req.example)
            return forecast_dataframe(
                df,
                target_column=req.target_column,
                id_column=req.id_column,
                feature_columns=req.feature_columns,
                risk_threshold=req.risk_threshold,
            )
        if req.csv_text:
            return forecast_csv(
                req.csv_text,
                target_column=req.target_column,
                id_column=req.id_column,
                feature_columns=req.feature_columns,
                risk_threshold=req.risk_threshold,
            )
        if req.rows:
            df = pd.DataFrame(req.rows)
            return forecast_dataframe(
                df,
                target_column=req.target_column,
                id_column=req.id_column,
                feature_columns=req.feature_columns,
                risk_threshold=req.risk_threshold,
            )
        raise ValueError("Provide 'example', 'csv_text', or 'rows'.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>PeakWhale™ Pilot</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 20px;">
  <h1>PeakWhale™ Pilot</h1>
  <p>Domain-agnostic outcome forecaster and early-warning engine. Explore the interactive API docs at <a href="/docs">/docs</a>.</p>
</body>
</html>"""
