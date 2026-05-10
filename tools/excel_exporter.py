"""Export analysis reports to Excel with multiple formatted sheets."""
import io
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def export_to_excel(
    report: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> bytes:
    """Build a multi-sheet Excel file from a report dict. Returns bytes."""
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wb = writer.book

        # ── Sheet 1: Summary ──────────────────────────────────────────
        summary_data = {
            "Field": [
                "Report Generated", "Query", "Rows Returned",
                "Anomalies Detected", "Trend Direction",
                "Executive Summary",
            ],
            "Value": [
                report.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
                report.get("query", ""),
                report.get("data", {}).get("row_count", 0),
                report.get("anomaly_count", 0),
                report.get("trends", {}).get("direction", "N/A").title(),
                report.get("executive_summary", ""),
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # ── Sheet 2: Raw Data ─────────────────────────────────────────
        rows = report.get("data", {}).get("rows", [])
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Data", index=False)
        elif df is not None and not df.empty:
            df.to_excel(writer, sheet_name="Data", index=False)

        # ── Sheet 3: Key Findings & Recommendations ───────────────────
        findings = report.get("key_findings", [])
        recs = report.get("recommendations", [])
        insights_rows = []
        for f in findings:
            insights_rows.append({"Type": "Finding", "Content": f, "Priority": ""})
        for r in recs:
            insights_rows.append({
                "Type": "Recommendation",
                "Content": r.get("action", ""),
                "Priority": r.get("priority", "").upper(),
            })
        if insights_rows:
            pd.DataFrame(insights_rows).to_excel(writer, sheet_name="Insights", index=False)

        # ── Sheet 4: Anomalies ────────────────────────────────────────
        anomaly_rows = report.get("anomalies", {}).get("anomaly_rows", [])
        if anomaly_rows:
            pd.DataFrame(anomaly_rows).to_excel(writer, sheet_name="Anomalies", index=False)

        # ── Sheet 5: Trend & Correlation ──────────────────────────────
        trend = report.get("trends", {})
        mom = trend.get("mom_changes", [])
        if mom:
            pd.DataFrame(mom).to_excel(writer, sheet_name="Trends", index=False)

        corr = report.get("correlations", {})
        top_pairs = corr.get("top_pairs", [])
        if top_pairs:
            pd.DataFrame(top_pairs).to_excel(writer, sheet_name="Correlations", index=False)

        # ── Sheet 6: SQL ──────────────────────────────────────────────
        sql_df = pd.DataFrame([{"SQL Query Used": report.get("sql", "")}])
        sql_df.to_excel(writer, sheet_name="SQL", index=False)

        # ── Auto-format column widths ─────────────────────────────────
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max((len(str(cell.value or "")) for cell in col), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    buf.seek(0)
    return buf.read()
