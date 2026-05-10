"""Renders a compiled report dict into Streamlit UI components."""
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
_FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "data" / "feedback.json"


def _save_feedback(query: str, vote: str) -> None:
    try:
        _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8")) if _FEEDBACK_FILE.exists() else []
        existing.append({"query": query[:200], "vote": vote, "ts": datetime.now().isoformat()})
        _FEEDBACK_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save feedback: %s", exc)


def _build_override_chart(rows: list, chart_type: str, title: str = "") -> dict:
    """Build an aggregated chart from raw rows for the View As picker."""
    from tools.viz_generator import bar_chart, line_chart, scatter_chart, table_chart
    if not rows:
        return {}

    df = pd.DataFrame(rows)

    # Numeric cols — skip ID/index-like columns, prefer business metrics
    _id_skip = {"id", "index", "row_num", "rownum"}
    _prefer  = ("total", "revenue", "sales", "amount", "profit", "price", "quantity", "count", "score")
    raw_num  = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in raw_num if c.lower() not in _id_skip and not c.lower().endswith("_id")]
    num_cols = sorted(num_cols, key=lambda c: next((i for i, k in enumerate(_prefer) if k in c.lower()), 99))
    if not num_cols:
        num_cols = raw_num

    # Categorical cols — skip high-cardinality / ID-like ones
    _cat_skip = {"customer_name", "name", "order_id", "email", "address", "description"}
    all_cat   = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols  = [c for c in all_cat if c.lower() not in _cat_skip and df[c].nunique() <= 30]

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "month", "year", "time", "week"))]

    def _total_label(label: str) -> str:
        """Return 'Total <label>' only when label doesn't already start with 'total'."""
        return label if label.lower().startswith("total") else f"Total {label}"

    try:
        metric = num_cols[0] if num_cols else None
        dim    = cat_cols[0]  if cat_cols  else None
        date   = date_cols[0] if date_cols else None
        m_label = metric.replace("_", " ").title() if metric else "Value"
        d_label = dim.replace("_", " ").title()    if dim    else ""

        if chart_type == "Bar":
            if dim and metric:
                agg = df.groupby(dim)[metric].sum().reset_index().nlargest(15, metric)
                return bar_chart(agg, x=dim, y=metric,
                                 title=title or f"{_total_label(m_label)} by {d_label}")
            elif metric:
                agg = df[[metric]].describe().reset_index()
                return table_chart(agg, title=f"{m_label} Summary")

        elif chart_type == "Line":
            if date and metric:
                tmp = df.copy()
                tmp[date] = pd.to_datetime(tmp[date], errors="coerce")
                tmp = tmp.dropna(subset=[date])
                agg = (tmp.groupby(tmp[date].dt.to_period("M").astype(str))[metric]
                         .sum().reset_index())
                agg.columns = [date, metric]
                agg = agg.sort_values(date)
                return line_chart(agg, x=date, y=metric,
                                  title=title or f"Monthly {m_label} Over Time")
            elif dim and metric:
                agg = df.groupby(dim)[metric].sum().reset_index()
                return line_chart(agg, x=dim, y=metric, title=title or f"{m_label} by {d_label}")

        elif chart_type == "Scatter":
            if len(num_cols) >= 2:
                sample = df[num_cols[:2]].dropna().head(300)
                return scatter_chart(sample, x=num_cols[0], y=num_cols[1],
                                     title=title or f"{num_cols[0].replace('_',' ').title()} vs {num_cols[1].replace('_',' ').title()}")
            elif dim and metric:
                sample = df[[dim, metric]].dropna().head(300)
                return scatter_chart(sample, x=dim, y=metric,
                                     title=title or f"{d_label} vs {m_label}")

        elif chart_type == "Table":
            if dim and metric:
                total_col = _total_label(m_label)
                agg = (df.groupby(dim)[metric].agg(["sum", "mean", "count"])
                         .reset_index()
                         .rename(columns={"sum": total_col, "mean": "Average", "count": "Orders"})
                         .sort_values(total_col, ascending=False)
                         .head(20))
                return table_chart(agg, title=f"{m_label} by {d_label}")
            elif metric:
                summary = df[num_cols[:4]].describe().round(2).reset_index()
                return table_chart(summary, title="Statistical Summary")

        # Fallback: bar if possible
        if dim and metric:
            agg = df.groupby(dim)[metric].sum().reset_index().nlargest(15, metric)
            return bar_chart(agg, x=dim, y=metric, title=title or f"{_total_label(m_label)} by {d_label}")

        return table_chart(df[num_cols[:5]].head(20) if num_cols else df.head(20), title=title)

    except Exception as exc:
        logger.warning("Chart override failed (%s): %s", chart_type, exc)
        return table_chart(df.head(20), title=title)


def _chart_caption(fig_dict: dict) -> str:
    """Return a plain-English explanation for a Plotly figure dict."""
    traces = fig_dict.get("data", [])
    if not traces:
        return ""
    chart_type = traces[0].get("type", "")
    mode = traces[0].get("mode", "")

    if chart_type == "scatter" and "lines" in mode:
        return "📈 **Line chart** — shows how a value changes over time. A rising line means growth; a falling line means decline. Look for sudden jumps or drops that need attention."
    if chart_type == "scatter" and "markers" in mode:
        return "🔵 **Scatter chart** — each dot represents one record. The position shows the relationship between two measures. Dots close together share similar values; spread-out dots mean little relationship."
    if chart_type == "bar":
        return "📊 **Bar chart** — compares values across categories. Taller bars mean higher values. Use this to quickly spot the top and bottom performers."
    if chart_type == "box":
        return "📦 **Box chart** — shows how values are distributed. The box covers the middle 50% of records. The line inside is the average. Dots outside the lines are unusual values worth investigating."
    if chart_type == "heatmap":
        return "🌡️ **Correlation heatmap** — shows how strongly pairs of columns relate to each other. **Dark blue** means they move together (both go up or both go down). **Dark red** means they move in opposite directions. Values close to 0 mean no relationship."
    if chart_type == "histogram":
        return "📉 **Histogram** — shows how often different value ranges appear in your data. A tall bar means many records fall in that range. Useful for spotting skewed distributions or unusual concentrations."
    if chart_type == "table":
        return "🗂️ **Data table** — a direct view of the records returned by your query."
    if chart_type == "indicator":
        return "🔢 **KPI indicator** — a single key number summarising the result at a glance."
    return ""


def render_report(report: dict[str, Any]) -> None:
    """Main renderer — takes a report dict and renders the full analysis UI."""
    if not report:
        return

    import hashlib
    _key = hashlib.md5(str(report.get("query", "") + str(id(report))).encode()).hexdigest()[:8]

    if report.get("status") == "error":
        st.error(f"Query failed: {report.get('error', 'Unknown error')}")
        _render_sql_block(report.get("sql", ""))
        return

    # ── Cleaning Report (shown instead of normal layout when cleaning ran) ──
    data = report.get("data", {})
    clean_report = data.get("clean_report") or (report.get("sql_result") or {}).get("clean_report")
    if not clean_report:
        # Also check nested inside data rows (compile_report stores sql_result rows)
        for row in data.get("rows", []):
            if isinstance(row, dict) and "clean_report" in row:
                clean_report = row["clean_report"]
                break
    if clean_report:
        _render_clean_report(clean_report, data)
        return

    # ── Executive Summary ──────────────────────────────────────────────
    summary = report.get("executive_summary", "")
    if summary:
        st.info(f"**Summary:** {summary}")

    # ── Anomaly Alert ──────────────────────────────────────────────────
    if report.get("anomaly_alert"):
        st.warning(
            f"⚠️ **{report['anomaly_count']} anomalies detected** across the dataset."
        )

    # ── Metrics Row ────────────────────────────────────────────────────
    data = report.get("data", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows Returned", f"{data.get('row_count', 0):,}")
    col2.metric("Columns", len(data.get("columns", [])))
    anomaly_count = report.get("anomaly_count", 0)
    col3.metric("Anomalies", anomaly_count, delta=None if anomaly_count == 0 else f"⚠️")
    trend = report.get("trends", {})
    col4.metric(
        "Trend",
        trend.get("direction", "N/A").title(),
        delta=f"{trend.get('direction_strength_pct', 0):.1f}%" if trend else None,
    )

    # ── Key Findings ───────────────────────────────────────────────────
    findings = report.get("key_findings", [])
    if findings:
        with st.expander("**Key Findings**", expanded=True):
            for f in findings:
                st.markdown(f"- {f}")

    # ── Chart type picker ─────────────────────────────────────────────
    viz_list = report.get("visualizations", [])
    rows = report.get("data", {}).get("rows", [])

    # Show picker whenever there is data — even if no auto chart was generated
    if viz_list or rows:
        st.subheader("Charts")
        chart_choice = st.radio(
            "View as:",
            ["Auto", "Bar", "Line", "Scatter", "Table"],
            horizontal=True,
            key=f"chart_type_{_key}",
        )

        if chart_choice == "Auto":
            display_figs = viz_list
        elif rows:
            display_figs = [_build_override_chart(rows, chart_choice, report.get("query", ""))]
        else:
            display_figs = viz_list

        if not display_figs:
            st.info("No chart available for **Auto** on this result. Select Bar, Line, Scatter, or Table to visualise the data.")
        else:
            cols = st.columns(min(len(display_figs), 2))
            for i, fig_dict in enumerate(display_figs):
                with cols[i % 2]:
                    try:
                        fig = go.Figure(fig_dict)
                        st.plotly_chart(fig, use_container_width=True, key=f"report_chart_{_key}_{i}")
                        caption = _chart_caption(fig_dict)
                        if caption:
                            st.caption(caption)
                    except Exception:
                        st.warning("Could not render chart.")

    # ── Recommendations ───────────────────────────────────────────────
    recommendations = report.get("recommendations", [])
    if recommendations:
        with st.expander("**Action Recommendations**", expanded=True):
            for rec in recommendations:
                priority = rec.get("priority", "medium").upper()
                color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
                st.markdown(
                    f"{color} **{rec.get('action', '')}**  \n"
                    f"*{rec.get('rationale', '')}*"
                )

    # ── Data Table ────────────────────────────────────────────────────
    rows = data.get("rows", [])
    if rows:
        with st.expander(f"Raw Data ({len(rows):,} rows)", expanded=False):
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

    # ── Trend Details ─────────────────────────────────────────────────
    if trend and not trend.get("error"):
        with st.expander("Trend Details", expanded=False):
            st.markdown(f"**Direction:** {trend.get('direction', '').title()}")
            yoy = trend.get("yoy_change")
            if yoy:
                sign = "+" if yoy["pct"] > 0 else ""
                st.markdown(f"**Year-over-Year:** {sign}{yoy['pct']:.1f}%")
            mom = trend.get("mom_changes", [])
            if mom:
                mom_df = pd.DataFrame(mom)
                st.dataframe(mom_df, use_container_width=True)

    # ── Anomaly Details ───────────────────────────────────────────────
    anomaly_data = report.get("anomalies", {})
    if anomaly_data and anomaly_data.get("total_anomalies", 0) > 0:
        with st.expander("Anomaly Details", expanded=False):
            st.markdown(anomaly_data.get("summary", ""))
            flagged = anomaly_data.get("anomaly_rows", [])
            if flagged:
                st.dataframe(pd.DataFrame(flagged[:20]), use_container_width=True)

    # ── Correlations ─────────────────────────────────────────────────
    corr = report.get("correlations", {})
    if corr and not corr.get("error"):
        with st.expander("Correlation Analysis", expanded=False):
            st.markdown(corr.get("summary", ""))
            top_pairs = corr.get("top_pairs", [])
            if top_pairs:
                st.dataframe(pd.DataFrame(top_pairs[:10]), use_container_width=True)

    # ── SQL Used ──────────────────────────────────────────────────────
    _render_sql_block(report.get("sql", ""))

    # ── Follow-up Questions ───────────────────────────────────────────
    follow_ups = report.get("follow_up_questions", [])
    if follow_ups:
        with st.expander("Suggested Follow-Up Questions", expanded=False):
            for q in follow_ups:
                st.markdown(f"- *{q}*")

    # ── Downloads + Save ──────────────────────────────────────────────
    st.markdown("**Export & Save**")
    dl1, dl2, dl3, dl4 = st.columns(4)

    # Markdown
    from tools.report_generator import report_to_markdown
    md = report_to_markdown(report)
    dl1.download_button("⬇️ Markdown", data=md,
                        file_name="report.md", mime="text/markdown",
                        use_container_width=True, key=f"dl_md_{_key}")

    # Excel
    try:
        from tools.excel_exporter import export_to_excel
        xlsx = export_to_excel(report)
        dl2.download_button("⬇️ Excel", data=xlsx,
                            file_name="report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key=f"dl_xlsx_{_key}")
    except Exception:
        pass

    # PDF
    try:
        from tools.pdf_exporter import export_to_pdf
        pdf_bytes = export_to_pdf(report)
        dl3.download_button("⬇️ PDF", data=pdf_bytes,
                            file_name="report.pdf", mime="application/pdf",
                            use_container_width=True, key=f"dl_pdf_{_key}")
    except Exception:
        pass

    # Save query to bookmarks
    if dl4.button("🔖 Save Query", use_container_width=True, key=f"dl_save_{_key}"):
        try:
            from app.main import _save_current_query
            _save_current_query(report.get("query", ""))
            st.success("Query saved!")
        except Exception:
            pass

    # ── Feedback ──────────────────────────────────────────────────────
    fb_state_key = f"feedback_{_key}"
    if st.session_state.get(fb_state_key) is None:
        st.markdown("**Was this answer helpful?**")
        fb1, fb2, _ = st.columns([1, 1, 8])
        if fb1.button("👍 Yes", key=f"fb_up_{_key}", use_container_width=True):
            st.session_state[fb_state_key] = "up"
            _save_feedback(report.get("query", ""), "positive")
            st.rerun()
        if fb2.button("👎 No", key=f"fb_down_{_key}", use_container_width=True):
            st.session_state[fb_state_key] = "down"
            _save_feedback(report.get("query", ""), "negative")
            st.rerun()
    else:
        fb = st.session_state[fb_state_key]
        st.caption("✅ Thanks for the feedback!" if fb == "up" else "📝 Noted — we'll keep improving.")


def _render_clean_report(clean_report: dict, data: dict) -> None:
    """Dedicated layout for data cleaning results."""
    st.success(f"**Data Cleaned Successfully**")
    st.markdown(f"_{clean_report.get('summary', '')}_")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", f"{clean_report.get('original_rows', 0):,}")
    c2.metric("Cleaned Rows", f"{clean_report.get('final_rows', 0):,}")
    c3.metric("Rows Removed", f"{clean_report.get('rows_removed', 0):,}")
    c4.metric("Operations", clean_report.get("total_steps", 0))

    # Steps
    steps = clean_report.get("steps", [])
    if steps:
        with st.expander("Cleaning Steps Applied", expanded=True):
            for i, step in enumerate(steps, 1):
                op = step.get("operation", "").replace("_", " ").title()
                detail = step.get("detail", "")
                st.markdown(f"**{i}. {op}** — {detail}")
                log = step.get("log") or step.get("examples") or []
                for entry in log[:5]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {entry}")

    # Preview + Download cleaned data
    clean_name = data.get("clean_table_name", "")
    source_table = data.get("source_table", "table")

    # Fetch full cleaned table from DuckDB for download
    full_df = None
    if clean_name:
        try:
            from utils.db_connector import run_query
            full_df = run_query(f'SELECT * FROM "{clean_name}"')
        except Exception:
            pass

    if full_df is not None and not full_df.empty:
        with st.expander("Preview Cleaned Data", expanded=True):
            st.dataframe(full_df.head(100), use_container_width=True)

        csv_bytes = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download Cleaned Data as CSV ({len(full_df):,} rows)",
            data=csv_bytes,
            file_name=f"{source_table}_cleaned.csv",
            mime="text/csv",
            type="primary",
        )
    elif data.get("rows"):
        with st.expander("Preview Cleaned Data", expanded=True):
            st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True)

    if clean_name:
        st.info(
            f"Cleaned data saved as table **`{clean_name}`** in the database. "
            f"Tick it in the sidebar under Loaded Tables to query it."
        )


def render_autonomous_alerts(alerts: list[dict]) -> None:
    st.subheader("Autonomous Scan Results")
    if not alerts:
        st.success("No significant anomalies detected in the current scan.")
        return

    for alert in alerts:
        with st.container():
            st.error(f"**{alert['table']}** — {alert['anomaly_count']} anomalies")
            st.markdown(alert.get("alert_text", ""))
            st.divider()


def _render_sql_block(sql: str) -> None:
    if sql:
        with st.expander("SQL Query Used", expanded=False):
            st.code(sql, language="sql")
