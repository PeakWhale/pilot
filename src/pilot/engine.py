"""Interpretable outcome forecasting and early-warning driver attribution engine."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


class FeatureWeight(BaseModel):
    feature: str
    coefficient: float
    standardized_importance: float
    mean_value: float
    direction: str


class RecordForecast(BaseModel):
    row_id: str
    actual: float | None
    predicted: float
    range_low: float
    range_high: float
    at_risk: bool
    top_drivers: list[str]


class ForecastResult(BaseModel):
    target_column: str
    id_column: str | None
    feature_columns: list[str]
    row_count: int
    cv_r2: float
    cv_mae: float
    interval_half_width: float
    risk_threshold: float
    at_risk_count: int
    weights: list[FeatureWeight]
    records: list[RecordForecast]
    headline: str


def _examples_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "examples").is_dir():
        return here / "examples"
    repo_ex = here.parent.parent / "examples"
    if repo_ex.is_dir():
        return repo_ex
    raise FileNotFoundError("Could not locate examples directory.")


def list_examples() -> list[str]:
    ex = _examples_dir()
    return sorted(p.stem for p in ex.glob("*.csv"))


def load_example(name: str) -> pd.DataFrame:
    ex = _examples_dir() / f"{name}.csv"
    if not ex.exists():
        available = ", ".join(list_examples())
        raise ValueError(f"Unknown example '{name}'. Available: {available}")
    return pd.read_csv(ex)


def forecast_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
    feature_columns: list[str] | None = None,
    risk_threshold: float | None = None,
) -> ForecastResult:
    if df.empty:
        raise ValueError("Input table is empty.")

    if id_column is None:
        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            id_column = str(non_num[0])
    elif id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in columns: {list(df.columns)}")

    num_cols = [
        str(c) for c in df.columns
        if c != id_column and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    if len(num_cols) < 2:
        raise ValueError("Need at least one numeric feature column and one numeric target column.")

    if target_column is None:
        target_column = num_cols[-1]
    elif target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in columns: {list(df.columns)}")

    if feature_columns is None:
        feature_columns = [c for c in num_cols if c != target_column]
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")

    if not feature_columns:
        raise ValueError("No feature columns left after excluding target_column.")

    work = df.dropna(subset=feature_columns + [target_column]).copy()
    n_rows = len(work)
    if n_rows < 8:
        raise ValueError(f"Need at least 8 complete rows to train and cross-validate (got {n_rows}).")

    X = work[feature_columns].to_numpy(dtype=float)
    y = work[target_column].to_numpy(dtype=float)

    n_splits = min(5, n_rows // 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    oof_preds = cross_val_predict(model, X, y, cv=kf)
    cv_r2 = float(r2_score(y, oof_preds))
    cv_mae = float(mean_absolute_error(y, oof_preds))
    residuals = np.abs(y - oof_preds)
    half_width = float(np.percentile(residuals, 80))

    model.fit(X, y)
    preds = model.predict(X)
    means = X.mean(axis=0)
    stds = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    coefs = model.coef_

    if risk_threshold is None:
        risk_threshold = round(float(np.percentile(y, 25)), 2)

    weights: list[FeatureWeight] = []
    for idx, col in enumerate(feature_columns):
        c_val = float(coefs[idx])
        std_imp = float(abs(c_val * stds[idx]))
        weights.append(
            FeatureWeight(
                feature=col,
                coefficient=round(c_val, 4),
                standardized_importance=round(std_imp, 4),
                mean_value=round(float(means[idx]), 3),
                direction="positive" if c_val >= 0 else "negative",
            )
        )
    weights.sort(key=lambda w: w.standardized_importance, reverse=True)

    records: list[RecordForecast] = []
    for idx_pos, (_, row) in enumerate(work.iterrows()):
        rid = str(row[id_column]) if id_column else f"row-{idx_pos + 1}"
        pred_val = float(preds[idx_pos])
        act_val = float(row[target_column])
        contribs = coefs * (X[idx_pos] - means)
        ranked_idx = sorted(range(len(feature_columns)), key=lambda i: abs(contribs[i]), reverse=True)
        top_drivers: list[str] = []
        for fi in ranked_idx[:3]:
            delta = float(contribs[fi])
            col = feature_columns[fi]
            val = float(X[idx_pos, fi])
            avg = float(means[fi])
            top_drivers.append(f"{delta:+.2f} from {col} ({val:g} vs avg {avg:.1f})")

        records.append(
            RecordForecast(
                row_id=rid,
                actual=round(act_val, 3),
                predicted=round(pred_val, 2),
                range_low=round(pred_val - half_width, 2),
                range_high=round(pred_val + half_width, 2),
                at_risk=bool(pred_val < risk_threshold),
                top_drivers=top_drivers,
            )
        )

    at_risk_count = sum(1 for r in records if r.at_risk)
    headline = (
        f"Forecasted '{target_column}' for {n_rows} records (held-out R2 {cv_r2:.2f}, MAE {cv_mae:.2f}); "
        f"{at_risk_count} record{'s' if at_risk_count != 1 else ''} flagged below threshold {risk_threshold:g}."
    )

    return ForecastResult(
        target_column=target_column,
        id_column=id_column,
        feature_columns=feature_columns,
        row_count=n_rows,
        cv_r2=round(cv_r2, 4),
        cv_mae=round(cv_mae, 4),
        interval_half_width=round(half_width, 3),
        risk_threshold=risk_threshold,
        at_risk_count=at_risk_count,
        weights=weights,
        records=records,
        headline=headline,
    )


def forecast_csv(
    csv_source: str | Path | io.StringIO,
    target_column: str | None = None,
    id_column: str | None = None,
    feature_columns: list[str] | None = None,
    risk_threshold: float | None = None,
) -> ForecastResult:
    if isinstance(csv_source, (str, Path)) and Path(csv_source).exists():
        df = pd.read_csv(csv_source)
    elif isinstance(csv_source, str):
        df = pd.read_csv(io.StringIO(csv_source))
    else:
        df = pd.read_csv(csv_source)
    return forecast_dataframe(
        df,
        target_column=target_column,
        id_column=id_column,
        feature_columns=feature_columns,
        risk_threshold=risk_threshold,
    )
