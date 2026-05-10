import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

TEMPLATE = "plotly_white"

# Vibrant, distinct palette for categorical series
COLORS = [
    "#6C63FF",  # purple-blue
    "#FF6B6B",  # coral red
    "#48CAE4",  # sky blue
    "#F9A825",  # amber
    "#06D6A0",  # mint green
    "#FF9F43",  # orange
    "#EE5A24",  # deep orange
    "#0652DD",  # royal blue
    "#C44569",  # rose
    "#A29BFE",  # lavender
]

# Gradient scales per chart type
_SEQ_SCALE   = "Blues"          # single-series histograms / bars
_DIV_SCALE   = "RdYlGn"         # diverging (e.g. profit/loss)
_CORR_SCALE  = "RdBu"           # correlation heatmap
_ANOM_NORMAL = "#48CAE4"        # normal points in anomaly chart
_ANOM_FLAG   = "#FF6B6B"        # anomalous points

# Columns that look like IDs — skip for charting
_ID_PATTERNS = ("id", "_id", "index", "row_num", "rownum")
# Preferred metric columns — pick these first when available
_METRIC_PRIORITY = ("total", "revenue", "sales", "amount", "profit", "income",
                    "spend", "cost", "price", "count", "quantity", "value", "score")


def _is_id_col(col: str, df: pd.DataFrame) -> bool:
    """Return True if the column looks like a meaningless row identifier."""
    name = col.lower()
    if name in _ID_PATTERNS or any(name.endswith(p) for p in _ID_PATTERNS):
        return True
    # Sequential integers with no repetition → likely an index
    s = df[col].dropna()
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 1:
        diffs = s.sort_values().diff().dropna().unique()
        if len(diffs) == 1:
            return True
    return False


def _best_numeric(cols: list[str], df: pd.DataFrame) -> list[str]:
    """Return numeric columns ordered by usefulness — skip ID-like columns."""
    filtered = [c for c in cols if not _is_id_col(c, df)]
    if not filtered:
        return cols  # fallback: use everything if all look like IDs

    def priority(col: str) -> int:
        name = col.lower()
        for i, kw in enumerate(_METRIC_PRIORITY):
            if kw in name:
                return i
        return len(_METRIC_PRIORITY)

    return sorted(filtered, key=priority)


def _fmt_col(col: str) -> str:
    """Turn snake_case / underscore names into readable Title Case labels."""
    return col.replace("_", " ").title()


def _apply_number_format(fig: go.Figure, axis: str = "y") -> go.Figure:
    """Format large axis numbers as 1K, 1M etc. for readability."""
    tickformat = ",.0f"
    if axis == "y":
        fig.update_yaxes(tickformat=tickformat)
    else:
        fig.update_xaxes(tickformat=tickformat)
    return fig


def _clean_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B"), x=0),
        template=TEMPLATE,
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#1E293B")),
        margin=dict(t=60, b=60, l=60, r=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8FAFC",
    )
    fig.update_xaxes(
        title_font=dict(color="#1E293B", size=13),
        tickfont=dict(color="#334155", size=12),
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
    )
    fig.update_yaxes(
        title_font=dict(color="#1E293B", size=13),
        tickfont=dict(color="#334155", size=12),
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
    )
    return fig


def auto_chart(df: pd.DataFrame, title: str = "") -> dict:
    """Pick the most meaningful chart type automatically."""
    if df.empty:
        return _empty_chart(title)

    raw_num = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df.columns
                 if any(kw in c.lower() for kw in ("date", "month", "week", "year", "time", "day"))]

    num_cols = _best_numeric(raw_num, df)

    # Time series — sort by date for a clean line
    if date_cols and num_cols:
        sorted_df = df.copy()
        try:
            sorted_df[date_cols[0]] = pd.to_datetime(sorted_df[date_cols[0]], errors="coerce")
            sorted_df = sorted_df.sort_values(date_cols[0])
        except Exception:
            pass
        return line_chart(sorted_df, x=date_cols[0], y=num_cols[0], title=title)

    # Category vs metric — top 15 to avoid clutter
    if cat_cols and num_cols:
        col_cat, col_num = cat_cols[0], num_cols[0]
        plot_df = df[[col_cat, col_num]].dropna()
        if plot_df[col_cat].nunique() > 15:
            plot_df = (
                plot_df.groupby(col_cat, as_index=False)[col_num]
                .sum()
                .nlargest(15, col_num)
            )
        return bar_chart(plot_df, x=col_cat, y=col_num, title=title)

    # Two numeric columns → scatter
    if len(num_cols) >= 2:
        return scatter_chart(df, x=num_cols[0], y=num_cols[1], title=title)

    # Single numeric → histogram
    if num_cols:
        return histogram_chart(df, col=num_cols[0], title=title)

    return table_chart(df, title=title)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    orientation: str = "v",
) -> dict:
    # Cap categories so the chart stays readable
    plot_df = df.copy()
    if plot_df[x].nunique() > 20:
        top = plot_df.groupby(x)[y].sum().nlargest(15).index
        plot_df = plot_df[plot_df[x].isin(top)]

    # Use multi-color when no explicit color grouping, else use palette
    if color is None and plot_df[x].nunique() > 1:
        bar_colors = COLORS
    else:
        bar_colors = COLORS

    fig = px.bar(
        plot_df, x=x, y=y, color=color if color else x,
        title=title,
        orientation=orientation,
        template=TEMPLATE,
        color_discrete_sequence=bar_colors,
        text_auto=".2s",
        labels={x: _fmt_col(x), y: _fmt_col(y)},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        opacity=0.9,
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(showlegend=False)
    _apply_number_format(fig, "y")
    _clean_layout(fig, title or f"{_fmt_col(y)} by {_fmt_col(x)}")
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

    for i, col in enumerate(y_cols):
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            mode="lines+markers",
            name=_fmt_col(col),
            line=dict(color=c, width=2.5),
            marker=dict(size=7, color=c, line=dict(width=1.5, color="white")),
        ))
        if show_moving_avg and len(df) >= 7:
            ma = df[col].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df[x], y=ma,
                mode="lines",
                name=f"{_fmt_col(col)} (7-day avg)",
                line=dict(dash="dash", color=c, width=1.5),
            ))

    fig.update_xaxes(tickangle=-30, title_text=_fmt_col(x))
    fig.update_yaxes(title_text=_fmt_col(y_cols[0]) if len(y_cols) == 1 else "Value")
    _apply_number_format(fig, "y")
    _clean_layout(fig, title or f"{_fmt_col(y_cols[0])} Over Time")
    return fig.to_dict()


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    title: str = "",
) -> dict:
    try:
        import statsmodels  # noqa: F401
        trendline = "ols"
    except ImportError:
        trendline = None

    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        template=TEMPLATE,
        color_discrete_sequence=COLORS,
        trendline=trendline,
        labels={x: _fmt_col(x), y: _fmt_col(y)},
        opacity=0.7,
    )
    _apply_number_format(fig, "y")
    _clean_layout(fig, title or f"{_fmt_col(x)} vs {_fmt_col(y)}")
    return fig.to_dict()


def heatmap_chart(matrix: dict[str, dict], title: str = "Correlation Matrix") -> dict:
    labels = list(matrix.keys())
    readable = [_fmt_col(l) for l in labels]
    values = [[matrix[r].get(c, 0) or 0 for c in labels] for r in labels]

    fig = go.Figure(go.Heatmap(
        z=values,
        x=readable,
        y=readable,
        colorscale=_CORR_SCALE,
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Strength", tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["−1", "−0.5", "0", "+0.5", "+1"]),
    ))
    _clean_layout(fig, title)
    return fig.to_dict()


def box_chart(df: pd.DataFrame, columns: list[str] | None = None, title: str = "") -> dict:
    raw = columns or df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in raw if not _is_id_col(c, df)][:6]
    if not cols:
        cols = raw[:6]

    fig = go.Figure()
    for i, col in enumerate(cols):
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Box(
            y=df[col],
            name=_fmt_col(col),
            marker=dict(color=c, size=5, outliercolor=c),
            line=dict(color=c, width=2),
            boxpoints="outliers",
            jitter=0.4,
        ))

    fig.update_yaxes(title_text="Value")
    _apply_number_format(fig, "y")
    _clean_layout(fig, title or "Value Distribution by Column")
    return fig.to_dict()


def anomaly_chart(
    df: pd.DataFrame,
    col: str,
    anomaly_indices: list[int],
    title: str = "",
) -> dict:
    anom_set = set(anomaly_indices)
    normal_x = [i for i in df.index if i not in anom_set]
    anom_x   = [i for i in df.index if i in anom_set]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_x, y=df.loc[normal_x, col] if normal_x else [],
        mode="markers", name="Normal",
        marker=dict(color=_ANOM_NORMAL, size=6, opacity=0.65,
                    line=dict(width=0.5, color="white")),
    ))
    fig.add_trace(go.Scatter(
        x=anom_x, y=df.loc[anom_x, col] if anom_x else [],
        mode="markers", name="⚠️ Anomaly",
        marker=dict(color=_ANOM_FLAG, size=11, symbol="x-thin",
                    line=dict(width=2.5, color=_ANOM_FLAG)),
    ))

    fig.update_xaxes(title_text="Record Index")
    fig.update_yaxes(title_text=_fmt_col(col))
    _apply_number_format(fig, "y")
    _clean_layout(fig, title or f"Anomalies in {_fmt_col(col)}")
    return fig.to_dict()


def histogram_chart(df: pd.DataFrame, col: str, title: str = "") -> dict:
    fig = px.histogram(
        df, x=col, nbins=30,
        template=TEMPLATE,
        color_discrete_sequence=["#6C63FF"],
        labels={col: _fmt_col(col)},
        opacity=1.0,
    )
    fig.update_traces(
        marker_color="#6C63FF",
        marker_line_width=1,
        marker_line_color="white",
    )
    _apply_number_format(fig, "x")
    _clean_layout(fig, title or f"Distribution of {_fmt_col(col)}")
    return fig.to_dict()


def table_chart(df: pd.DataFrame, title: str = "", max_rows: int = 50) -> dict:
    preview = df.head(max_rows)
    readable_cols = [_fmt_col(c) for c in preview.columns]
    n_rows = len(preview)
    n_cols = len(preview.columns)
    row_colors = ["#1E293B" if i % 2 == 0 else "#0F172A" for i in range(n_rows)]
    cell_colors = [list(row_colors) for _ in range(n_cols)]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in readable_cols],
            fill_color="#4361EE",
            font=dict(color="#FFFFFF", size=13),
            align="left",
            height=38,
            line=dict(color="#2D3FBF", width=1),
        ),
        cells=dict(
            values=[preview[col].astype(str).tolist() for col in preview.columns],
            align="left",
            height=32,
            fill_color=cell_colors,
            font=dict(color="#F1F5F9", size=12),
            line=dict(color="#334155", width=1),
        ),
    ))
    fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A")
    fig.update_layout(title=title, margin=dict(t=40, b=10))
    return fig.to_dict()


def _empty_chart(title: str) -> dict:
    fig = go.Figure()
    fig.update_layout(
        title=title or "No data",
        annotations=[dict(text="No data to display", showarrow=False,
                          font=dict(size=14, color="gray"))],
        template=TEMPLATE,
    )
    return fig.to_dict()


def charts_from_report(report: dict) -> list[dict]:
    return report.get("visualizations", [])
