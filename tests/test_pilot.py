"""Tests for PeakWhale™ Pilot forecasting engine, CLI, and HTTP API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from pilot.api import app
from pilot.cli import main as cli_main
from pilot.engine import forecast_dataframe, list_examples, load_example


def test_examples_load_and_forecast():
    examples = list_examples()
    assert "account-health" in examples
    assert "program-outcomes" in examples

    df = load_example("account-health")
    res = forecast_dataframe(df, risk_threshold=65.0)
    assert res.target_column == "renewal_health_score"
    assert res.row_count == 44
    assert res.cv_r2 > 0.85
    assert res.at_risk_count > 0
    assert len(res.records) == 44
    assert all(len(r.top_drivers) > 0 for r in res.records)


def test_cli_csv_roundtrip(tmp_path):
    out_csv = tmp_path / "forecasts.csv"
    rc = cli_main(["forecast", "--example", "program-outcomes", "--threshold", "60", "--out-csv", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    txt = out_csv.read_text()
    assert "top_drivers" in txt
    assert "PART-301" in txt


def test_api_endpoints():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "peakwhale-pilot"

    r2 = client.post("/api/forecast", json={"example": "account-health", "risk_threshold": 60.0})
    assert r2.status_code == 200
    body = r2.json()
    assert body["target_column"] == "renewal_health_score"
    assert body["cv_r2"] > 0.85
