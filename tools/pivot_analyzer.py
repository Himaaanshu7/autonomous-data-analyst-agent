"""Pivot table / crosstab analysis."""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_pivot(
    df: pd.DataFrame,
    row_col: str | None = None,
    col_col: str | None = None,
    value_col: str | None = None,
    aggfunc: str = "sum",
) -> dict[str, Any]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(cat_cols) < 1:
        return {"error": "Need at least one categorical column for pivot table."}
    if len(num_cols) < 1:
        return {"error": "Need at least one numeric column for pivot table."}

    row_col = row_col or cat_cols[0]
    col_col = col_col or (cat_cols[1] if len(cat_cols) > 1 else None)
    value_col = value_col or num_cols[0]

    agg_map = {"sum": "sum", "mean": "mean", "count": "count",
               "max": "max", "min": "min", "median": "median"}
    fn = agg_map.get(aggfunc, "sum")

    if col_col and col_col != row_col:
        pivot = pd.pivot_table(
            df, values=value_col, index=row_col,
            columns=col_col, aggfunc=fn, fill_value=0,
        )
        pivot.columns = [str(c) for c in pivot.columns]
    else:
        pivot = df.groupby(row_col)[value_col].agg(fn).reset_index()
        pivot = pivot.set_index(row_col)

    pivot = pivot.round(2)

    # Top insights
    flat = pivot.stack().reset_index() if isinstance(pivot.columns, pd.Index) and len(pivot.columns) > 1 else pivot
    max_val = pivot.values.max()
    min_val = pivot.values.min()

    return {
        "row_col": row_col,
        "col_col": col_col,
        "value_col": value_col,
        "aggfunc": fn,
        "table": pivot.reset_index().to_dict(orient="records"),
        "columns": [row_col] + list(pivot.columns),
        "max_value": float(max_val),
        "min_value": float(min_val),
        "summary": (
            f"Pivot: {value_col} ({fn}) by {row_col}"
            + (f" × {col_col}" if col_col and col_col != row_col else "")
            + f". Max={max_val:,.2f}, Min={min_val:,.2f}."
        ),
    }


def pivot_chart(result: dict) -> dict:
    import plotly.express as px
    rows = result.get("table", [])
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    row_col = result.get("row_col", df.columns[0])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return {}
    col_col = result.get("col_col")
    if col_col and col_col in df.columns:
        fig = px.bar(df, x=row_col, y=num_cols[0], color=col_col,
                     title=result.get("summary", "Pivot Table"),
                     template="plotly_white", barmode="group")
    else:
        fig = px.bar(df, x=row_col, y=num_cols[0],
                     title=result.get("summary", "Pivot Table"),
                     template="plotly_white")
    return fig.to_dict()
