"""LangGraph-powered Autonomous Data Analyst Agent.

Graph topology:
    START → plan → execute_sql → [reflect | analyze] → visualize → insights → report → END
                        ↑               |
                        └── reflect ────┘  (max 3 retries)
"""
import logging
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from tools.sql_generator import generate_sql, fix_sql
from tools.sql_executor import execute_sql
from tools.data_profiler import profile_dataframe
from tools.anomaly_detector import detect_anomalies
from tools.trend_detector import detect_trends
from tools.correlation_analyzer import analyze_correlations
from tools.viz_generator import auto_chart, bar_chart, line_chart, box_chart, heatmap_chart
from tools.insight_generator import generate_insights
from tools.report_generator import compile_report
from utils.llm_client import llm_client
from utils.schema_inspector import get_all_schemas_as_text
from utils.memory_manager import memory

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: plan
# ---------------------------------------------------------------------------

def plan_node(state: AgentState) -> dict[str, Any]:
    """LLM decides which tools are needed and in what order."""
    schema_text = state.get("schema_text") or get_all_schemas_as_text()
    history = memory.get_history_text()

    user_prompt = build_planner_prompt(state["query"], schema_text, history)
    try:
        plan = llm_client.complete_json(
            system=PLANNER_SYSTEM_PROMPT,
            user=user_prompt,
            cache_system=False,
        )
    except (ValueError, Exception) as exc:
        logger.warning("Planner failed, using default plan: %s", exc)
        plan = {
            "query_type": "exploratory",
            "needs_sql": True,
            "needs_time_analysis": False,
            "needs_anomaly_detection": False,
            "needs_correlation": False,
            "chart_types": ["bar"],
        }

    logger.info("Plan: %s", plan)
    return {"plan": plan, "schema_text": schema_text, "retry_count": 0}


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql_node(state: AgentState) -> dict[str, Any]:
    """Generate SQL if needed, then execute it."""
    sql = state.get("sql", "")
    if not sql:
        sql = generate_sql(state["query"], state.get("schema_text"))

    result = execute_sql(sql)
    return {
        "sql": sql,
        "sql_result": result.to_dict(),
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# Node: reflect  (retry with fixed SQL)
# ---------------------------------------------------------------------------

def reflect_node(state: AgentState) -> dict[str, Any]:
    """Ask the model to fix the failing SQL query."""
    schema_text = state.get("schema_text") or get_all_schemas_as_text()
    fixed_sql = fix_sql(
        user_query=state["query"],
        failed_sql=state.get("sql", ""),
        error_message=state.get("error", ""),
        schema_text=schema_text,
    )
    return {
        "sql": fixed_sql,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: analyze
# ---------------------------------------------------------------------------

def analyze_node(state: AgentState) -> dict[str, Any]:
    """Run statistical analysis on the SQL result."""
    sql_result = state.get("sql_result", {})
    if not sql_result.get("success"):
        return {}

    df = pd.DataFrame(sql_result.get("data", []))
    if df.empty:
        return {"profile": {}, "anomalies": {}, "trends": {}, "correlations": {}}

    plan = state.get("plan", {})
    updates: dict[str, Any] = {}

    # Always profile
    updates["profile"] = profile_dataframe(df)

    # Anomaly detection
    if plan.get("needs_anomaly_detection") or "anomal" in state["query"].lower():
        updates["anomalies"] = detect_anomalies(df)

    # Trend detection
    if plan.get("needs_time_analysis") or any(
        kw in state["query"].lower() for kw in ("trend", "over time", "monthly", "weekly", "growth", "drop", "decline")
    ):
        updates["trends"] = detect_trends(df)

    # Correlation
    if plan.get("needs_correlation") or "correlat" in state["query"].lower():
        updates["correlations"] = analyze_correlations(df)

    return updates


# ---------------------------------------------------------------------------
# Node: visualize
# ---------------------------------------------------------------------------

def visualize_node(state: AgentState) -> dict[str, Any]:
    """Generate Plotly chart dicts from the analysis results."""
    sql_result = state.get("sql_result", {})
    if not sql_result.get("success"):
        return {"visualizations": []}

    df = pd.DataFrame(sql_result.get("data", []))
    if df.empty:
        return {"visualizations": []}

    plan = state.get("plan", {})
    chart_types = plan.get("chart_types", ["auto"])
    query = state["query"].lower()
    charts: list[dict] = []

    # Primary chart
    primary = auto_chart(df, title=f"Results: {state['query'][:60]}")
    charts.append(primary)

    # Box plot for distributions
    if "distribution" in query or "spread" in query or state.get("profile"):
        charts.append(box_chart(df, title="Distribution Overview"))

    # Anomaly chart
    anomalies = state.get("anomalies", {})
    if anomalies and anomalies.get("total_anomalies", 0) > 0:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col, info in anomalies.get("columns", {}).items():
            if col in df.columns and info.get("total", 0) > 0:
                from tools.viz_generator import anomaly_chart
                charts.append(
                    anomaly_chart(df, col, info["anomaly_indices"], title=f"Anomalies: {col}")
                )
                break

    # Correlation heatmap
    correlations = state.get("correlations", {})
    if correlations and not correlations.get("error"):
        charts.append(heatmap_chart(correlations["matrix"], title="Correlation Heatmap"))

    return {"visualizations": charts}


# ---------------------------------------------------------------------------
# Node: insights
# ---------------------------------------------------------------------------

def insights_node(state: AgentState) -> dict[str, Any]:
    """Ask the LLM to generate business insights from all analysis."""
    sql_result = state.get("sql_result", {})
    if not sql_result.get("success"):
        return {"insights": {"summary": state.get("error", "Query failed."), "key_findings": [], "recommendations": []}}

    insights = generate_insights(
        user_query=state["query"],
        df_dict=sql_result,
        anomaly_result=state.get("anomalies"),
        trend_result=state.get("trends"),
    )
    return {"insights": insights}


# ---------------------------------------------------------------------------
# Node: report
# ---------------------------------------------------------------------------

def report_node(state: AgentState) -> dict[str, Any]:
    """Compile everything into the final report dict."""
    report = compile_report(
        user_query=state["query"],
        sql=state.get("sql", ""),
        sql_result=state.get("sql_result", {"success": False, "error": state.get("error", "")}),
        profile=state.get("profile"),
        anomalies=state.get("anomalies"),
        trends=state.get("trends"),
        correlations=state.get("correlations"),
        insights=state.get("insights"),
        visualizations=state.get("visualizations"),
    )

    memory.add_query(
        query=state["query"],
        result_summary=report.get("executive_summary", "")[:200],
    )

    return {
        "report": report,
        "messages": [{"role": "assistant", "content": report.get("executive_summary", "")}],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_execute(
    state: AgentState,
) -> Literal["reflect", "analyze"]:
    if state.get("error") and state.get("retry_count", 0) < 3:
        return "reflect"
    return "analyze"


def route_after_reflect(
    state: AgentState,
) -> Literal["execute_sql", "report"]:
    if state.get("retry_count", 0) >= 3:
        logger.error("Max retries reached. Giving up on SQL.")
        return "report"
    return "execute_sql"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_agent() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("visualize", visualize_node)
    graph.add_node("insights", insights_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute_sql")

    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {"reflect": "reflect", "analyze": "analyze"},
    )
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"execute_sql": "execute_sql", "report": "report"},
    )

    graph.add_edge("analyze", "visualize")
    graph.add_edge("visualize", "insights")
    graph.add_edge("insights", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Singleton compiled agent
analyst_agent = build_agent()


def run_query(user_query: str) -> dict[str, Any]:
    """Public entry point — invoke the agent and return the report."""
    initial_state: AgentState = {
        "query": user_query,
        "schema_text": "",
        "plan": {},
        "sql": "",
        "sql_result": {},
        "profile": {},
        "anomalies": {},
        "trends": {},
        "correlations": {},
        "visualizations": [],
        "insights": {},
        "report": {},
        "error": "",
        "retry_count": 0,
        "messages": [{"role": "user", "content": user_query}],
    }
    final_state = analyst_agent.invoke(initial_state)
    return final_state.get("report", {})
