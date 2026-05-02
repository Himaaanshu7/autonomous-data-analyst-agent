import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

COLORS = px.colors.qualitative.Plotly
TEMPLATE = "plotly_white"


def auto_chart(df: pd.DataFrame, title: str = "") -> dict:
    """Choose the most appropriate chart type automatically."""
    if df.empty:
        return _empty_chart(title)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower() or "year" in c.lower()]

    if date_cols and numeric_cols:
        return line_chart(df, x=date_cols[0], y=numeric_cols[0], title=title)
    if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
        return bar_chart(df, x=categorical_cols[0], y=numeric_cols[0], title=title)
    if len(numeric_cols) >= 2:
        return scatter_chart(df, x=numeric_cols[0], y=numeric_cols[1], title=title)
    if categorical_cols and numeric_cols:
        return bar_chart(df, x=categorical_cols[0], y=numeric_cols[0], title=title)
    return table_chart(df, title=title)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    orientation: str = "v",
) -> dict:
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        orientation=orientation,
        template=TEMPLATE,
        color_discrete_sequence=COLORS,
    )
    fig.update_layout(xaxis_tickangle=-30, margin=dict(t=50, b=50))
    return fig.to_dict()


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    color: str | None = None,
    title: str = "",
    show_moving_avg: bool = False,
) -> dict:
    y_cols = [y] if isinstance(y, str) else y
    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=col))
        if show_moving_avg and len(df) >= 7:
            ma = df[col].rolling(7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=ma,
                    mode="lines",
                    name=f"{col} (7-day MA)",
                    line=dict(dash="dash"),
                )
            )

    fig.update_layout(title=title, template=TEMPLATE, xaxis_tickangle=-30)
    return fig.to_dict()


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    title: str = "",
) -> dict:
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        title=title, template=TEMPLATE,
        color_discrete_sequence=COLORS,
        trendline="ols",
    )
    return fig.to_dict()


def heatmap_chart(matrix: dict[str, dict], title: str = "Correlation Matrix") -> dict:
    """Render a correlation matrix as a heatmap."""
    labels = list(matrix.keys())
    values = [[matrix[r].get(c, 0) or 0 for c in labels] for r in labels]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in values],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=title, template=TEMPLATE)
    return fig.to_dict()


def box_chart(df: pd.DataFrame, columns: list[str] | None = None, title: str = "") -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    fig = go.Figure()
    for col in cols[:6]:
        fig.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers"))
    fig.update_layout(title=title or "Distribution Overview", template=TEMPLATE)
    return fig.to_dict()


def anomaly_chart(
    df: pd.DataFrame,
    col: str,
    anomaly_indices: list[int],
    title: str = "",
) -> dict:
    """Scatter plot with anomalous points highlighted in red."""
    colors = ["red" if i in set(anomaly_indices) else "steelblue" for i in df.index]
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(df))),
            y=df[col],
            mode="markers",
            marker=dict(color=colors, size=6),
            text=[
                f"Anomaly: {v}" if i in set(anomaly_indices) else str(v)
                for i, v in enumerate(df[col])
            ],
        )
    )
    fig.update_layout(
        title=title or f"Anomalies in '{col}'",
        xaxis_title="Index",
        yaxis_title=col,
        template=TEMPLATE,
    )
    return fig.to_dict()


def table_chart(df: pd.DataFrame, title: str = "", max_rows: int = 50) -> dict:
    preview = df.head(max_rows)
    fig = go.Figure(
        go.Table(
            header=dict(
                values=list(preview.columns),
                fill_color="steelblue",
                font_color="white",
                align="left",
            ),
            cells=dict(
                values=[preview[col].tolist() for col in preview.columns],
                align="left",
            ),
        )
    )
    fig.update_layout(title=title, margin=dict(t=40, b=10))
    return fig.to_dict()


def _empty_chart(title: str) -> dict:
    fig = go.Figure()
    fig.update_layout(
        title=title or "No data",
        annotations=[dict(text="No data to display", showarrow=False)],
    )
    return fig.to_dict()


def charts_from_report(report: dict) -> list[dict]:
    """Extract all Plotly figure dicts from a report."""
    return report.get("visualizations", [])
