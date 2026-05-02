import logging
from typing import Any

from agents.prompts import INSIGHT_SYSTEM_PROMPT, build_insight_prompt
from utils.llm_client import llm_client
from tools.data_profiler import profile_to_text

logger = logging.getLogger(__name__)


def generate_insights(
    user_query: str,
    df_dict: dict[str, Any],
    anomaly_result: dict[str, Any] | None = None,
    trend_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the LLM to produce business insights and recommendations.

    df_dict is the sql_result dict from QueryResult.to_dict().
    """
    # Build a readable data summary from the result
    data_summary = _build_data_summary(df_dict)
    anomaly_summary = anomaly_result.get("summary", "") if anomaly_result else ""
    trend_summary = trend_result.get("summary", "") if trend_result else ""

    user_prompt = build_insight_prompt(
        user_query=user_query,
        data_summary=data_summary,
        anomaly_summary=anomaly_summary,
        trend_summary=trend_summary,
    )

    try:
        result = llm_client.complete_json(
            system=INSIGHT_SYSTEM_PROMPT,
            user=user_prompt,
            cache_system=False,
        )
    except ValueError as exc:
        logger.warning("Insight JSON parse failed, using raw text: %s", exc)
        raw = llm_client.complete(system=INSIGHT_SYSTEM_PROMPT, user=user_prompt)
        result = {
            "summary": raw[:500],
            "key_findings": [],
            "anomalies_explained": [],
            "recommendations": [],
            "follow_up_questions": [],
        }

    return result


def _build_data_summary(df_dict: dict) -> str:
    if not df_dict.get("success"):
        return f"Query failed: {df_dict.get('error', 'unknown error')}"

    rows = df_dict.get("data", [])
    columns = df_dict.get("columns", [])
    row_count = df_dict.get("row_count", 0)

    lines = [
        f"Columns: {', '.join(columns)}",
        f"Total rows: {row_count:,}",
        "",
        "First 10 rows:",
    ]
    for r in rows[:10]:
        lines.append("  " + str(r))

    stats = df_dict.get("data_summary", "")
    if stats:
        lines += ["", "Statistical Summary:", stats[:1000]]

    return "\n".join(lines)
