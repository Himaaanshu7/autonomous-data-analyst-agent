"""Time-series forecasting using linear regression + moving average."""
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def forecast(
    df: pd.DataFrame,
    date_col: str | None = None,
    value_col: str | None = None,
    periods: int = 6,
) -> dict[str, Any]:
    date_col = date_col or _find_date_col(df)
    value_col = value_col or _find_value_col(df, date_col)

    if not date_col or not value_col:
        return {"error": "Could not identify date and value columns for forecasting."}

    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna().sort_values(date_col)

    if len(ts) < 4:
        return {"error": "Not enough data points to forecast (need at least 4)."}

    # Aggregate to monthly
    ts = ts.set_index(date_col).resample("ME")[value_col].sum().reset_index()

    x = np.arange(len(ts))
    y = ts[value_col].values
    slope, intercept, r, p, se = stats.linregress(x, y)

    # Forecast future periods
    future_x = np.arange(len(ts), len(ts) + periods)
    forecast_vals = slope * future_x + intercept

    # Confidence interval (95%)
    t_val = stats.t.ppf(0.975, df=len(ts) - 2)
    residuals = y - (slope * x + intercept)
    rmse = np.sqrt(np.mean(residuals ** 2))
    ci = t_val * rmse

    # Last known date → generate future dates
    last_date = ts[date_col].iloc[-1]
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="ME")[1:]

    historical = [
        {"date": str(row[date_col].date()), "value": round(float(row[value_col]), 2),
         "type": "historical"}
        for _, row in ts.iterrows()
    ]
    forecast_points = [
        {
            "date": str(d.date()),
            "value": round(float(v), 2),
            "lower": round(float(v - ci), 2),
            "upper": round(float(v + ci), 2),
            "type": "forecast",
        }
        for d, v in zip(future_dates, forecast_vals)
    ]

    trend_dir = "upward" if slope > 0 else "downward"
    pct_change = round((forecast_vals[-1] - y[-1]) / abs(y[-1]) * 100, 1) if y[-1] != 0 else 0

    return {
        "date_col": date_col,
        "value_col": value_col,
        "historical": historical,
        "forecast": forecast_points,
        "model": {"slope": round(float(slope), 4), "r_squared": round(float(r ** 2), 4)},
        "trend": trend_dir,
        "forecast_periods": periods,
        "pct_change_projected": pct_change,
        "summary": (
            f"Forecast {periods} months ahead. Trend is {trend_dir}. "
            f"Projected {'+' if pct_change >= 0 else ''}{pct_change}% change. "
            f"Model R²={round(r**2, 3)}."
        ),
    }


def forecast_chart(result: dict) -> dict:
    import plotly.graph_objects as go
    fig = go.Figure()

    hist = result.get("historical", [])
    fcast = result.get("forecast", [])

    if hist:
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in hist], y=[p["value"] for p in hist],
            mode="lines+markers", name="Historical", line=dict(color="steelblue"),
        ))

    if fcast:
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in fcast], y=[p["value"] for p in fcast],
            mode="lines+markers", name="Forecast",
            line=dict(color="orange", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in fcast] + [p["date"] for p in reversed(fcast)],
            y=[p["upper"] for p in fcast] + [p["lower"] for p in reversed(fcast)],
            fill="toself", fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    fig.update_layout(
        title=f"Forecast: {result.get('value_col', '')} — next {result.get('forecast_periods', 6)} months",
        template="plotly_white", xaxis_title="Date", yaxis_title=result.get("value_col", ""),
    )
    return fig.to_dict()


def _find_date_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "month", "time", "period", "year")):
            return c
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() / len(df) > 0.8:
                return c
        except Exception:
            pass
    return None


def _find_value_col(df, exclude):
    for kw in ("revenue", "sales", "amount", "total", "value", "price", "income"):
        for c in df.select_dtypes(include="number").columns:
            if c != exclude and kw in c.lower():
                return c
    nums = [c for c in df.select_dtypes(include="number").columns if c != exclude]
    return nums[0] if nums else None
