"""Cohort retention analysis — tracks groups of users/customers over time."""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cohort_analysis(
    df: pd.DataFrame,
    user_col: str | None = None,
    date_col: str | None = None,
    event_col: str | None = None,
) -> dict[str, Any]:
    user_col = user_col or _find_col(df, ("customer", "user", "id", "client", "member"))
    date_col = date_col or _find_date_col(df)

    if not user_col or not date_col:
        return {
            "error": (
                "Cohort analysis requires a user/customer ID column and a date column. "
                f"Available columns: {', '.join(df.columns.tolist())}"
            )
        }

    work = df[[user_col, date_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna()
    work["cohort_month"] = work.groupby(user_col)[date_col].transform("min").dt.to_period("M")
    work["activity_month"] = work[date_col].dt.to_period("M")
    work["period_number"] = (
        work["activity_month"].astype(int) - work["cohort_month"].astype(int)
    )

    cohort_data = work.groupby(["cohort_month", "period_number"])[user_col].nunique().reset_index()
    cohort_data.columns = ["cohort_month", "period_number", "users"]

    cohort_sizes = cohort_data[cohort_data["period_number"] == 0].set_index("cohort_month")["users"]
    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period_number", values="users"
    )

    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0).round(3) * 100
    retention_matrix = retention_matrix.iloc[:, :12]  # cap at 12 periods

    cohorts = retention_matrix.index.astype(str).tolist()
    periods = [f"Month {i}" for i in retention_matrix.columns]

    avg_retention = retention_matrix.mean(skipna=True).round(1).to_dict()
    period_0_users = cohort_sizes.sum()

    return {
        "user_col": user_col,
        "date_col": date_col,
        "cohorts": cohorts,
        "periods": periods,
        "matrix": retention_matrix.fillna(0).values.tolist(),
        "avg_retention_by_period": {f"Month {k}": float(v) for k, v in avg_retention.items()},
        "total_users": int(period_0_users),
        "num_cohorts": len(cohorts),
        "summary": (
            f"Cohort analysis across {len(cohorts)} cohorts, {int(period_0_users):,} total users. "
            f"Month-1 avg retention: {avg_retention.get(1, 0):.1f}%."
        ),
    }


def cohort_chart(result: dict) -> dict:
    import plotly.graph_objects as go
    matrix = result.get("matrix", [])
    cohorts = result.get("cohorts", [])
    periods = result.get("periods", [])
    if not matrix:
        return {}
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=periods,
        y=cohorts,
        colorscale="Blues",
        text=[[f"{v:.1f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        zmin=0,
        zmax=100,
    ))
    fig.update_layout(
        title="Cohort Retention Analysis (%)",
        xaxis_title="Period",
        yaxis_title="Cohort",
        template="plotly_white",
    )
    return fig.to_dict()


def _find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


def _find_date_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "month", "created", "joined")):
            return c
    return None
