"""MCP server for PeakWhale™ Pilot."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from .engine import forecast_csv, forecast_dataframe, list_examples, load_example

mcp = FastMCP("peakwhale-pilot")


@mcp.tool()
def pilot_list_examples() -> list[str]:
    """List built-in synthetic datasets available in PeakWhale™ Pilot."""
    return list_examples()


@mcp.tool()
def pilot_forecast_example(example: str = "account-health", risk_threshold: float | None = None) -> dict:
    """Forecast outcomes and flag at-risk records on a built-in example dataset."""
    df = load_example(example)
    return forecast_dataframe(df, risk_threshold=risk_threshold).model_dump()


@mcp.tool()
def pilot_forecast_csv(csv_text: str, target_column: str | None = None, risk_threshold: float | None = None) -> dict:
    """Forecast outcomes on raw CSV text and return driver attributions and at-risk flags."""
    return forecast_csv(csv_text, target_column=target_column, risk_threshold=risk_threshold).model_dump()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
