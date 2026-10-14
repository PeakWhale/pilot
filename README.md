<p><img src="pilot.svg" width="64" height="64" alt="PeakWhale™ Pilot logo"></p>

# PeakWhale™ Pilot
### Forecast outcomes and flag at-risk records early, with plain-English drivers behind every score.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20Regression-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-HTTP%20API-teal)
![MCP](https://img.shields.io/badge/MCP-Server-informational)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)
[![CI](https://github.com/PeakWhale/pilot/actions/workflows/ci.yml/badge.svg)](https://github.com/PeakWhale/pilot/actions/workflows/ci.yml)

**PeakWhale™ Pilot** is a domain-agnostic outcome forecaster and early-warning engine. Feed it a CSV table where early activity signals lead to a final outcome, such as customer renewal health, project delivery readiness, or training program completion. Pilot fits an interpretable linear model with cross-validated accuracy metrics ($R^2$, MAE, and 80% prediction ranges), flags at-risk records before they fail, and explains the top positive and negative drivers for every single row.

> [!NOTE]
> All bundled example datasets (`account-health.csv` and `program-outcomes.csv`) are 100% synthetic. The original 2019 regression script is preserved in [`archive/`](archive/) for historical provenance.

## Quick start

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PeakWhale/pilot.git
cd pilot
uv sync
```

Run a built-in example from the command line:

```bash
uv run pilot forecast --example account-health
uv run pilot forecast --example program-outcomes --threshold 60 --out-csv forecasts.csv
```

Forecast on your own CSV file:

```bash
uv run pilot forecast path/to/your-data.csv --target renewal_score --threshold 65
```

## HTTP API and MCP server

Start the local FastAPI server (interactive OpenAPI docs at `http://127.0.0.1:8000/docs`):

```bash
uv run pilot serve
```

Or run the Model Context Protocol (MCP) server for Claude Desktop, Cursor, or any MCP client:

```bash
uv run pilot-mcp
```

## Tests

```bash
uv run pytest
```

## What it enables businesses to do

* **Customer retention and renewal early warning:** Spot accounts whose onboarding pace, seat activity, or open escalations put their renewal at risk weeks before renewal date.
* **Program and cohort completion tracking:** Identify participants or projects falling behind early checkpoints and see which specific lever will move their projected outcome most.
* **Transparent, auditable scoring:** Give operations teams a score they can verify, complete with held-out error ranges and per-record factor attributions in plain English.

## Connect and customize

Want help adapting **PeakWhale™ Pilot** to your own data, workflows, or infrastructure?

* **Book 30 minutes:** [Schedule a working session with Addy](https://adrkv.github.io/meet.html?audience=teams)
* **Portfolio and whitepapers:** [adrkv.github.io](https://adrkv.github.io)
* **LinkedIn:** [Addy Kaveti](https://www.linkedin.com/in/adrkv/) · [PeakWhale™](https://www.linkedin.com/company/peakwhale%E2%84%A2/)

## Part of the PeakWhale™ fleet

PeakWhale™ is an open-source initiative by [Addy Kaveti](https://adrkv.github.io) making practical AI and machine learning accessible to every team:

* **[PeakWhale™ Helm](https://github.com/PeakWhale/helm)**: local-first research assistant over your own SQL databases and documents, with source citations.
* **[PeakWhale™ Orca](https://github.com/PeakWhale/orca)**: self-hosted payment fraud check that scores transactions and explains every decision in plain English.
* **[PeakWhale™ Harbor](https://github.com/PeakWhale/harbor)**: turns any CSV spreadsheet into a prediction web app and JSON API with honest prediction ranges.
* **[PeakWhale™ Trident](https://github.com/PeakWhale/trident)**: fail-safe triage engine pairing AI with editable YAML rule packs so unclear cases always go to a person.
* **[PeakWhale™ Berth](https://github.com/PeakWhale/berth)**: EV depot fleet charging planner that schedules every vehicle to leave ready at the lowest power cost.
* **[PeakWhale™ Pod](https://github.com/PeakWhale/pod)**: domain-agnostic K-means cohort and regional clustering engine with plain-English segment profiles.
* **[PeakWhale™ Pilot](https://github.com/PeakWhale/pilot)**: outcome forecaster and early-warning engine that flags at-risk records and explains the top drivers.

## Legal, trademark, and liability notice

* **Open-source license:** Licensed under the **Apache License 2.0** (© 2019-2026 Aditya Rao (Addy) Kaveti). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). If you distribute, modify, or build on this software, you must retain the [`NOTICE`](NOTICE) file as required by Section 4(d) of the Apache License 2.0.
* **Trademark reservation:** **PeakWhale™** and the PeakWhale project marks and logos are trademarks of Aditya Rao (Addy) Kaveti. Under Section 6 of the Apache License 2.0, this license grants no permission to use the PeakWhale™ name, marks, or logos to endorse or promote derivative products or services without prior written permission.
* **No warranty or professional advice:** This software is provided **"AS IS"**, without warranty or liability of any kind (see Sections 7 and 8 of the Apache License 2.0). Output from this tool does not constitute financial, medical, legal, compliance, or safety advice. You are solely responsible for independent validation, security controls, human oversight, and regulatory compliance before using this software in production.
* **Independent personal project and synthetic data:** Created and maintained by Aditya Rao (Addy) Kaveti in an independent personal capacity. All bundled demo datasets are synthetic or public benchmark examples and contain no proprietary or employer material.
