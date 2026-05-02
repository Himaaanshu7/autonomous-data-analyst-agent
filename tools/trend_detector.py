import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_trends(
    df: pd.DataFrame,
    date_col: str | None = None,
    value_col: str | None = None,
    window: int = 7,
) -> dict[str, Any]:
    """Detect time-series trends: moving avg, MoM change, YoY change, direction."""

    # Auto-detect columns if not provided
    if date_col is None:
        date_col = _find_date_column(df)
    if value_col is None:
        value_col = _find_primary_numeric(df, exclude=date_col)

    if date_col is None or value_col is None:
        return {"error": "Could not identify date and value columns for trend analysis."}

    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna().sort_values(date_col)

    if len(ts) < 3:
        return {"error": "Not enough data points for trend detection."}

    ts["moving_avg"] = ts[value_col].rolling(window=min(window, len(ts)), min_periods=1).mean()

    # Overall direction via linear regression
    x = np.arange(len(ts))
    slope, intercept = np.polyfit(x, ts[value_col].fillna(0), 1)
    direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
    direction_strength = abs(slope) / (ts[value_col].mean() or 1) * 100

    # Period-over-period changes (monthly aggregation)
    ts_monthly = (
        ts.set_index(date_col)
        .resample("ME")[value_col]
        .sum()
        .reset_index()
    )

    mom_changes = []
    if len(ts_monthly) >= 2:
        ts_monthly["mom_change_pct"] = ts_monthly[value_col].pct_change() * 100
        mom_changes = ts_monthly.dropna(subset=["mom_change_pct"]).tail(6).to_dict(orient="records")

    yoy_change: dict | None = None
    if len(ts_monthly) >= 13:
        latest = ts_monthly.iloc[-1][value_col]
        year_ago = ts_monthly.iloc[-13][value_col]
        if year_ago != 0:
            yoy_pct = (latest - year_ago) / abs(year_ago) * 100
            yoy_change = {"pct": round(yoy_pct, 2), "latest": latest, "year_ago": year_ago}

    return {
        "date_col": date_col,
        "value_col": value_col,
        "direction": direction,
        "direction_strength_pct": round(direction_strength, 2),
        "slope": round(float(slope), 6),
        "window_days": window,
        "moving_avg_series": ts[["moving_avg"]].rename(
            columns={"moving_avg": "value"}
        ).assign(date=ts[date_col].values).to_dict(orient="records"),
        "mom_changes": [
            {k: (round(v, 4) if isinstance(v, float) else str(v)) for k, v in r.items()}
            for r in mom_changes
        ],
        "yoy_change": yoy_change,
        "summary": _build_summary(direction, direction_strength, yoy_change, mom_changes, value_col),
    }


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "month" in col.lower():
            return col
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() / len(df) > 0.8:
                return col
        except Exception:
            pass
    return None


def _find_primary_numeric(df: pd.DataFrame, exclude: str | None = None) -> str | None:
    candidates = [
        c for c in df.select_dtypes(include=[float, int]).columns
        if c != exclude
    ]
    # Prefer revenue/sales/amount columns
    for keyword in ("revenue", "sales", "amount", "total", "price", "value"):
        for col in candidates:
            if keyword in col.lower():
                return col
    return candidates[0] if candidates else None


def _build_summary(direction, strength, yoy, mom_changes, value_col) -> str:
    lines = [f"Trend in '{value_col}': {direction} (strength: {strength:.1f}%)"]
    if yoy:
        sign = "+" if yoy["pct"] > 0 else ""
        lines.append(f"Year-over-year change: {sign}{yoy['pct']:.1f}%")
    if mom_changes:
        last = mom_changes[-1]
        pct = last.get("mom_change_pct", 0)
        sign = "+" if pct > 0 else ""
        lines.append(f"Most recent month-over-month: {sign}{pct:.1f}%")
    return " | ".join(lines)
