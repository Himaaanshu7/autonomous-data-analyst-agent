"""Renders a compiled report dict into Streamlit UI components."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Any


def render_report(report: dict[str, Any]) -> None:
    """Main renderer — takes a report dict and renders the full analysis UI."""
    if not report:
        return

    if report.get("status") == "error":
        st.error(f"Query failed: {report.get('error', 'Unknown error')}")
        _render_sql_block(report.get("sql", ""))
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

    # ── Visualizations ────────────────────────────────────────────────
    viz_list = report.get("visualizations", [])
    if viz_list:
        st.subheader("Charts")
        cols = st.columns(min(len(viz_list), 2))
        for i, fig_dict in enumerate(viz_list):
            with cols[i % 2]:
                try:
                    fig = go.Figure(fig_dict)
                    st.plotly_chart(fig, use_container_width=True)
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

    # ── Download Report ───────────────────────────────────────────────
    from tools.report_generator import report_to_markdown
    md = report_to_markdown(report)
    st.download_button(
        label="Download Report (Markdown)",
        data=md,
        file_name="analysis_report.md",
        mime="text/markdown",
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
