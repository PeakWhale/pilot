"""Command-line interface for PeakWhale™ Pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .engine import forecast_csv, forecast_dataframe, list_examples, load_example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pilot",
        description="PeakWhale™ Pilot: forecast outcomes and flag at-risk records early with plain-English drivers.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_fc = sub.add_parser("forecast", help="Run outcome forecasting on a CSV file or built-in example")
    p_fc.add_argument("csv_path", nargs="?", help="Path to CSV file")
    p_fc.add_argument("--example", choices=list_examples(), help="Run a built-in synthetic example")
    p_fc.add_argument("--target", default=None, help="Target numeric column to forecast (defaults to last numeric column)")
    p_fc.add_argument("--id-column", default=None, help="Row identifier column")
    p_fc.add_argument("--threshold", type=float, default=None, help="At-risk threshold (defaults to 25th percentile)")
    p_fc.add_argument("--out-csv", default=None, help="Write per-record forecasts and drivers to CSV")
    p_fc.add_argument("--json", action="store_true", help="Print full result as JSON")

    p_serve = sub.add_parser("serve", help="Run the local HTTP API and web interface")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        uvicorn.run("pilot.api:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.cmd != "forecast":
        parser.print_help()
        return 1

    if args.example:
        df = load_example(args.example)
        res = forecast_dataframe(df, target_column=args.target, id_column=args.id_column, risk_threshold=args.threshold)
    elif args.csv_path:
        res = forecast_csv(Path(args.csv_path), target_column=args.target, id_column=args.id_column, risk_threshold=args.threshold)
    else:
        p_fc.error("Provide a CSV path or --example.")
        return 2

    if args.out_csv:
        import pandas as pd
        rows = [
            {
                "row_id": r.row_id,
                "actual": r.actual,
                "predicted": r.predicted,
                "range_low": r.range_low,
                "range_high": r.range_high,
                "at_risk": r.at_risk,
                "top_drivers": "; ".join(r.top_drivers),
            }
            for r in res.records
        ]
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
    else:
        print(res.headline)
        print("\nTop model signals:")
        for w in res.weights[:4]:
            print(f"  - {w.feature}: coef {w.coefficient:+.3f} ({w.direction}, importance {w.standardized_importance:.2f})")
        at_risk = [r for r in res.records if r.at_risk]
        if at_risk:
            print(f"\nSample at-risk records (predicted < {res.risk_threshold:g}):")
            for r in at_risk[:5]:
                print(f"  * {r.row_id}: predicted {r.predicted:.1f} [{r.range_low:.1f}..{r.range_high:.1f}] | {r.top_drivers[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
