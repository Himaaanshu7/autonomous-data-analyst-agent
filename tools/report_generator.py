import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def compile_report(
    user_query: str,
    sql: str,
    sql_result: dict[str, Any],
    profile: dict[str, Any] | None = None,
    anomalies: dict[str, Any] | None = None,
    trends: dict[str, Any] | None = None,
    correlations: dict[str, Any] | None = None,
    insights: dict[str, Any] | None = None,
    visualizations: list[dict] | None = None,
) -> dict[str, Any]:
    """Assemble every analysis output into a single structured report dict.

    The Streamlit UI consumes this dict directly.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report: dict[str, Any] = {
        "generated_at": now,
        "query": user_query,
        "status": "success" if sql_result.get("success") else "error",
        "sql": sql,
        "data": {
            "columns": sql_result.get("columns", []),
            "row_count": sql_result.get("row_count", 0),
            "rows": sql_result.get("data", []),
        },
        "profile": profile or {},
        "anomalies": anomalies or {},
        "trends": trends or {},
        "correlations": correlations or {},
        "insights": insights or {},
        "visualizations": visualizations or [],
        "error": sql_result.get("error", ""),
    }

    # Top-level summary card for the UI header
    if insights:
        report["executive_summary"] = insights.get("summary", "")
        report["key_findings"] = insights.get("key_findings", [])
        report["recommendations"] = insights.get("recommendations", [])
        report["follow_up_questions"] = insights.get("follow_up_questions", [])
    else:
        report["executive_summary"] = (
            f"Query returned {sql_result.get('row_count', 0):,} rows."
            if sql_result.get("success")
            else f"Query failed: {sql_result.get('error', '')}"
        )
        report["key_findings"] = []
        report["recommendations"] = []
        report["follow_up_questions"] = []

    # Anomaly alert (used by autonomous mode)
    anomaly_count = (anomalies or {}).get("total_anomalies", 0)
    report["anomaly_alert"] = anomaly_count > 0
    report["anomaly_count"] = anomaly_count

    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    """Convert a report dict to a markdown string (for download / email)."""
    lines = [
        f"# Data Analysis Report",
        f"**Generated:** {report['generated_at']}",
        f"**Query:** {report['query']}",
        "",
        "## Executive Summary",
        report.get("executive_summary", "N/A"),
        "",
    ]

    if report.get("key_findings"):
        lines += ["## Key Findings"]
        for f in report["key_findings"]:
            lines.append(f"- {f}")
        lines.append("")

    if report.get("recommendations"):
        lines += ["## Recommendations"]
        for r in report["recommendations"]:
            priority = r.get("priority", "medium").upper()
            lines.append(f"- **[{priority}]** {r.get('action', '')} — {r.get('rationale', '')}")
        lines.append("")

    if report.get("anomaly_count", 0) > 0:
        lines += [
            "## Anomaly Report",
            f"Total anomalies detected: **{report['anomaly_count']}**",
            report.get("anomalies", {}).get("summary", ""),
            "",
        ]

    if report.get("trends"):
        t = report["trends"]
        lines += ["## Trend Analysis", t.get("summary", ""), ""]

    lines += [
        "## SQL Query Used",
        f"```sql\n{report.get('sql', 'N/A')}\n```",
    ]

    return "\n".join(lines)
