"""KPI Dashboard tab — always-on metrics for loaded tables."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Any


def render_kpi_dashboard(active_tables: list[str]) -> None:
    if not active_tables:
        st.info("Load a dataset first to see the KPI Dashboard.")
        return

    from utils.db_connector import run_query
    from tools.anomaly_detector import detect_anomalies
    from tools.trend_detector import detect_trends

    # Cache per table so heavy computations don't re-run on every rerun
    cache_key = "kpi_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    for table in active_tables:
        st.subheader(f"Table: `{table}`")
        try:
            df = run_query(f'SELECT * FROM "{table}" LIMIT 5000')
        except Exception as e:
            st.error(f"Could not load '{table}': {e}")
            continue

        # Use cached computations to avoid blocking every rerun
        cached = st.session_state[cache_key].get(table)
        if not cached:
            anom   = detect_anomalies(df) if df.select_dtypes(include="number").shape[1] > 0 else {}
            trend  = detect_trends(df)
            st.session_state[cache_key][table] = {"anom": anom, "trend": trend}
            cached = st.session_state[cache_key][table]

        anom  = cached["anom"]
        trend = cached["trend"]

        # ── Top metrics ──────────────────────────────────────────────
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
        null_pct  = round(df.isna().mean().mean() * 100, 1)
        dup_count = df.duplicated().sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric cols", len(num_cols))
        c4.metric("Null %", f"{null_pct}%", delta="⚠️" if null_pct > 10 else None)
        c5.metric("Duplicates", f"{int(dup_count):,}", delta="⚠️" if dup_count > 0 else None)

        col_left, col_right = st.columns(2)

        # ── Anomaly count ────────────────────────────────────────────
        with col_left:
            if num_cols:
                total_anom = anom.get("total_anomalies", 0)
                color = "#FF6B6B" if total_anom > 0 else "#06D6A0"
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=total_anom,
                    title={"text": "Anomalies Detected"},
                    delta={"reference": 0, "increasing": {"color": "red"}},
                    number={"font": {"color": color}},
                ))
                fig.update_layout(height=200, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"kpi_anom_{table}")
                if total_anom == 0:
                    st.caption("✅ No unusual values found — your data looks clean.")
                else:
                    st.caption(f"⚠️ **{total_anom} unusual values** were detected. These records fall far outside the normal range and may be errors, fraud, or genuine outliers worth reviewing.")

        # ── Trend direction ──────────────────────────────────────────
        with col_right:
            direction = trend.get("direction", "unknown")
            strength  = trend.get("direction_strength_pct", 0)
            arrow = {"upward": "↑", "downward": "↓", "flat": "→"}.get(direction, "–")
            color = {"upward": "#06D6A0", "downward": "#FF6B6B", "flat": "#F9A825"}.get(direction, "#A29BFE")
            fig = go.Figure(go.Indicator(
                mode="number",
                value=strength,
                number={"suffix": "%", "font": {"color": color}},
                title={"text": f"Trend Strength ({arrow} {direction.title()})"},
            ))
            fig.update_layout(height=200, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"kpi_trend_{table}")
            trend_captions = {
                "upward":   "📈 The overall direction of your data is **upward** — values are growing over time.",
                "downward": "📉 The overall direction of your data is **downward** — values are declining over time.",
                "flat":     "➡️ Your data is relatively **stable** — no strong growth or decline detected.",
            }
            st.caption(trend_captions.get(direction, "Trend direction could not be determined."))

        # ── Distribution of best numeric column ─────────────────────
        _skip = {"id", "index", "row_num", "rownum"}
        _prefer = ("total", "revenue", "sales", "amount", "profit", "price", "quantity")
        good_num_cols = [c for c in num_cols if c.lower() not in _skip and not c.lower().endswith("_id")]
        good_num_cols = sorted(good_num_cols, key=lambda c: next((i for i, k in enumerate(_prefer) if k in c.lower()), 99))
        if good_num_cols:
            col = good_num_cols[0]
            col_label = col.replace("_", " ").title()
            fig = px.histogram(
                df, x=col, nbins=30,
                title=f"Distribution of {col_label}",
                template="plotly_dark",
                color_discrete_sequence=["#6C63FF"],
                labels={col: col_label},
                opacity=0.88,
            )
            fig.update_traces(
                marker_color="#6C63FF",
                marker_line_width=1,
                marker_line_color="white",
            )
            fig.update_layout(height=280, margin=dict(t=50, b=40))
            st.plotly_chart(fig, use_container_width=True, key=f"kpi_hist_{table}_{col}")
            st.caption(f"📉 **Distribution of {col_label}** — each bar shows how many records fall within that value range. A tall bar means many records share similar values. A long tail to one side may indicate outliers or skewed data.")

        # ── Top values of first categorical column ───────────────────
        if cat_cols:
            col = cat_cols[0]
            col_label = col.replace("_", " ").title()
            top = df[col].value_counts().head(10).reset_index()
            top.columns = [col, "count"]
            fig = px.bar(
                top, x=col, y="count",
                title=f"Top {col_label} by Count",
                template="plotly_dark",
                color=col,
                color_discrete_sequence=["#6C63FF","#FF6B6B","#48CAE4","#F9A825",
                                         "#06D6A0","#FF9F43","#EE5A24","#0652DD",
                                         "#C44569","#A29BFE"],
                text_auto=True,
                labels={col: col_label, "count": "Count"},
            )
            fig.update_traces(textposition="outside", marker_line_width=0, opacity=0.9)
            fig.update_xaxes(tickangle=-30)
            fig.update_layout(height=280, margin=dict(t=50, b=60), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"kpi_bar_{table}_{col}")
            st.caption(f"📊 **Top {col_label} by frequency** — shows which {col_label.lower()} values appear most often in your data. The tallest bar is the most common. Numbers on top are the exact counts.")

        st.divider()
