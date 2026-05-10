"""LangGraph-powered Autonomous Data Analyst Agent.

Graph topology:
    START → plan → [clean | execute_sql] → [reflect | analyze] → visualize → insights → report → END
                              ↑                    |
                              └──── reflect ───────┘  (max 3 retries)
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
from tools.data_cleaner import clean_dataframe, save_cleaned_table
from tools.forecaster import forecast, forecast_chart
from tools.pivot_analyzer import build_pivot, pivot_chart
from tools.statistical_tester import run_statistical_tests
from tools.validator import validate
from tools.cohort_analyzer import cohort_analysis, cohort_chart
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
# Node: clean  (data cleaning path)
# ---------------------------------------------------------------------------

_CLEAN_KEYWORDS = (
    "clean", "cleaning", "fix missing", "fill missing", "remove duplicate",
    "drop duplicate", "fix data", "handle null", "handle missing",
    "prepare data", "preprocess", "data quality", "fix column",
    "standardize", "normalise", "normalize", "cap outlier",
)


def is_cleaning_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _CLEAN_KEYWORDS)


def clean_node(state: AgentState) -> dict[str, Any]:
    """Load the active table, clean it, save as <table>_cleaned, report results."""
    from utils.db_connector import list_tables, run_query
    from utils.schema_inspector import get_schema_for_tables

    query = state["query"].lower()
    tables = list_tables()
    if not tables:
        return {
            "sql_result": {"success": False, "error": "No tables loaded."},
            "error": "No tables loaded.",
        }

    # Pick the table: prefer one mentioned by name in the query
    target = next((t for t in tables if t in query), tables[0])

    try:
        df = run_query(f'SELECT * FROM "{target}"')
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}

    # Parse user preferences
    cap_outliers = any(kw in query for kw in ("cap outlier", "cap outl", "remove outlier"))

    cleaned_df, clean_report = clean_dataframe(
        df,
        drop_duplicates=True,
        fill_missing=True,
        strip_strings=True,
        fix_dtypes=True,
        drop_all_null_cols=True,
        cap_outliers=cap_outliers,
    )

    # Save cleaned table back to DuckDB
    clean_table_name = save_cleaned_table(cleaned_df, target)

    # Refresh schema in memory with the new table
    all_tables = list_tables()
    new_schema = get_schema_for_tables(all_tables)

    sql_result = {
        "success": True,
        "sql": f"-- Auto-cleaned '{target}' → saved as '{clean_table_name}'",
        "columns": list(cleaned_df.columns),
        "row_count": len(cleaned_df),
        "data": cleaned_df.head(20).to_dict(orient="records"),
        "data_summary": cleaned_df.describe(include="all").to_string(),
        "clean_report": clean_report,
        "clean_table_name": clean_table_name,
        "source_table": target,
    }

    return {
        "sql": sql_result["sql"],
        "sql_result": sql_result,
        "error": "",
        "schema_text": new_schema,
    }


# ---------------------------------------------------------------------------
# Node: execute_sql
# ---------------------------------------------------------------------------

def execute_sql_node(state: AgentState) -> dict[str, Any]:
    """Generate SQL if needed, then execute it.

    For vague queries (insights / overview / profile) fall back to
    SELECT * LIMIT 500 on the first available table so the analysis
    tools always have data to work with.
    """
    from utils.db_connector import list_tables

    sql = state.get("sql", "")
    if not sql:
        # Detect vague intent — skip complex SQL, just fetch a sample
        vague_keywords = ("insight", "overview", "profile", "analyse", "analyze this",
                          "tell me about", "what can you", "explore", "summarize")
        query_lower = state["query"].lower()
        if any(kw in query_lower for kw in vague_keywords):
            tables = list_tables()
            if tables:
                sql = f'SELECT * FROM "{tables[0]}" LIMIT 500'
            else:
                sql = "SELECT 1"
        else:
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
    query = state["query"].lower()
    updates: dict[str, Any] = {}

    # Always profile — safe on any DataFrame shape
    try:
        updates["profile"] = profile_dataframe(df)
    except Exception:
        updates["profile"] = {}

    # Anomaly detection — only if numeric columns exist
    if df.select_dtypes(include="number").shape[1] > 0:
        if plan.get("needs_anomaly_detection") or "anomal" in query:
            try:
                updates["anomalies"] = detect_anomalies(df)
            except Exception:
                pass

    # Trend detection — only if a date-like column exists
    if plan.get("needs_time_analysis") or any(
        kw in query for kw in ("trend", "over time", "monthly", "weekly", "growth", "drop", "decline")
    ):
        try:
            updates["trends"] = detect_trends(df)
        except Exception:
            pass

    # Correlation — only if 2+ numeric columns
    if df.select_dtypes(include="number").shape[1] >= 2:
        if plan.get("needs_correlation") or "correlat" in query:
            try:
                updates["correlations"] = analyze_correlations(df)
            except Exception:
                pass

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
    query = state["query"].lower()
    charts: list[dict] = []

    # Primary chart — always attempt
    try:
        primary = auto_chart(df, title=f"Results: {state['query'][:60]}")
        charts.append(primary)
    except Exception:
        pass

    # Box plot only if numeric columns exist
    has_numeric = df.select_dtypes(include="number").shape[1] > 0
    if has_numeric and ("distribution" in query or "spread" in query or state.get("profile")):
        try:
            charts.append(box_chart(df, title="Distribution Overview"))
        except Exception:
            pass

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

    # SQL failed — build a helpful error with available columns
    if not sql_result.get("success"):
        schema_text = state.get("schema_text", "")
        available = _extract_column_hints(schema_text)
        summary = (
            f"I could not run that query on your dataset. "
            f"{('Available columns: ' + available) if available else ''} "
            f"Try rephrasing with column names from your data."
        )
        return {"insights": {
            "summary": summary,
            "key_findings": [],
            "anomalies_explained": [],
            "recommendations": [{"action": "Rephrase your question using actual column names from the dataset.", "rationale": summary, "priority": "high"}],
            "follow_up_questions": [],
        }}

    # 0 rows returned — tell the user clearly
    if sql_result.get("row_count", 0) == 0:
        return {"insights": {
            "summary": "The query ran successfully but returned 0 rows. Try broadening your filters or check if the data exists.",
            "key_findings": ["No data matched the query criteria."],
            "anomalies_explained": [],
            "recommendations": [{"action": "Check column names and filter values", "rationale": "Query returned empty result", "priority": "medium"}],
            "follow_up_questions": ["What data is actually in this dataset?", "Show me all rows in this table."],
        }}

    try:
        insights = generate_insights(
            user_query=state["query"],
            df_dict=sql_result,
            anomaly_result=state.get("anomalies"),
            trend_result=state.get("trends"),
        )
    except Exception as exc:
        logger.warning("Insight generation failed: %s", exc)
        insights = {
            "summary": f"Analysis complete. {sql_result.get('row_count', 0):,} rows returned.",
            "key_findings": [],
            "anomalies_explained": [],
            "recommendations": [],
            "follow_up_questions": [],
        }
    return {"insights": insights}


def _extract_column_hints(schema_text: str) -> str:
    """Pull column names out of schema text for helpful error messages."""
    import re
    match = re.search(r"COLUMNS:\s*(.+)", schema_text)
    if match:
        return match.group(1)[:200]
    return ""


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
# Node: forecast
# ---------------------------------------------------------------------------

def forecast_node(state: AgentState) -> dict[str, Any]:
    from utils.db_connector import list_tables, run_query
    tables = list_tables()
    if not tables:
        return {"sql_result": {"success": False, "error": "No tables loaded."}, "error": "No tables loaded."}
    query = state["query"].lower()
    target = next((t for t in tables if t in query), tables[0])
    try:
        df = run_query(f'SELECT * FROM "{target}"')
        result = forecast(df, periods=6)
        chart = forecast_chart(result) if not result.get("error") else {}
        sql_result = {
            "success": not bool(result.get("error")),
            "sql": f"-- Forecast on '{target}'",
            "columns": ["date", "value", "type"],
            "row_count": len(result.get("historical", [])) + len(result.get("forecast", [])),
            "data": result.get("historical", []) + result.get("forecast", []),
            "forecast_result": result,
            "error": result.get("error", ""),
        }
        viz = [chart] if chart else []
        return {"sql": sql_result["sql"], "sql_result": sql_result, "error": result.get("error", ""), "visualizations": viz}
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: pivot
# ---------------------------------------------------------------------------

def pivot_node(state: AgentState) -> dict[str, Any]:
    from utils.db_connector import list_tables, run_query
    tables = list_tables()
    if not tables:
        return {"sql_result": {"success": False, "error": "No tables loaded."}, "error": "No tables loaded."}
    query = state["query"].lower()
    target = next((t for t in tables if t in query), tables[0])
    try:
        df = run_query(f'SELECT * FROM "{target}"')
        result = build_pivot(df)
        chart = pivot_chart(result) if not result.get("error") else {}
        sql_result = {
            "success": not bool(result.get("error")),
            "sql": f"-- Pivot on '{target}'",
            "columns": result.get("columns", []),
            "row_count": len(result.get("table", [])),
            "data": result.get("table", []),
            "pivot_result": result,
            "error": result.get("error", ""),
        }
        viz = [chart] if chart else []
        return {"sql": sql_result["sql"], "sql_result": sql_result, "error": result.get("error", ""), "visualizations": viz}
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: stat_test
# ---------------------------------------------------------------------------

def stat_test_node(state: AgentState) -> dict[str, Any]:
    from utils.db_connector import list_tables, run_query
    tables = list_tables()
    if not tables:
        return {"sql_result": {"success": False, "error": "No tables loaded."}, "error": "No tables loaded."}
    query = state["query"].lower()
    target = next((t for t in tables if t in query), tables[0])
    try:
        df = run_query(f'SELECT * FROM "{target}"')
        result = run_statistical_tests(df)
        sql_result = {
            "success": not bool(result.get("error")),
            "sql": f"-- Statistical tests on '{target}'",
            "columns": ["test", "statistic", "p_value", "significant"],
            "row_count": len(result.get("tests", [])),
            "data": result.get("tests", []),
            "stat_result": result,
            "error": result.get("error", ""),
        }
        return {"sql": sql_result["sql"], "sql_result": sql_result, "error": result.get("error", "")}
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: validate
# ---------------------------------------------------------------------------

def validate_node(state: AgentState) -> dict[str, Any]:
    from utils.db_connector import list_tables, run_query
    tables = list_tables()
    if not tables:
        return {"sql_result": {"success": False, "error": "No tables loaded."}, "error": "No tables loaded."}
    query = state["query"].lower()
    target = next((t for t in tables if t in query), tables[0])
    try:
        df = run_query(f'SELECT * FROM "{target}"')
        result = validate(df)
        sql_result = {
            "success": True,
            "sql": f"-- Validation on '{target}'",
            "columns": ["rule", "status", "violations", "pct"],
            "row_count": len(result.get("results", [])),
            "data": result.get("results", []),
            "validation_result": result,
        }
        return {"sql": sql_result["sql"], "sql_result": sql_result, "error": ""}
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Node: cohort
# ---------------------------------------------------------------------------

def cohort_node(state: AgentState) -> dict[str, Any]:
    from utils.db_connector import list_tables, run_query
    tables = list_tables()
    if not tables:
        return {"sql_result": {"success": False, "error": "No tables loaded."}, "error": "No tables loaded."}
    query = state["query"].lower()
    target = next((t for t in tables if t in query), tables[0])
    try:
        df = run_query(f'SELECT * FROM "{target}"')
        result = cohort_analysis(df)
        chart = cohort_chart(result) if not result.get("error") else {}
        sql_result = {
            "success": not bool(result.get("error")),
            "sql": f"-- Cohort analysis on '{target}'",
            "columns": result.get("periods", []),
            "row_count": result.get("num_cohorts", 0),
            "data": [{"cohort": c, **dict(zip(result.get("periods", []), row))}
                     for c, row in zip(result.get("cohorts", []), result.get("matrix", []))],
            "cohort_result": result,
            "error": result.get("error", ""),
        }
        viz = [chart] if chart else []
        return {"sql": sql_result["sql"], "sql_result": sql_result, "error": result.get("error", ""), "visualizations": viz}
    except Exception as exc:
        return {"sql_result": {"success": False, "error": str(exc)}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_FORECAST_KW   = ("forecast", "predict", "next month", "next quarter", "future", "projection", "predict")
_PIVOT_KW      = ("pivot", "crosstab", "cross tab", "breakdown by", "by region and", "by category and")
_STAT_KW       = ("significant", "t-test", "anova", "statistically", "p-value", "compare groups", "is the difference")
_VALIDATE_KW   = ("validate", "validation", "data quality", "check rules", "business rules", "flag violations")
_COHORT_KW     = ("cohort", "retention", "returning users", "returning customers", "churn")


def route_after_plan(state: AgentState) -> str:
    q = state.get("query", "").lower()
    if is_cleaning_query(q):            return "clean"
    if any(k in q for k in _FORECAST_KW):  return "forecast"
    if any(k in q for k in _PIVOT_KW):     return "pivot"
    if any(k in q for k in _STAT_KW):      return "stat_test"
    if any(k in q for k in _VALIDATE_KW):  return "validate"
    if any(k in q for k in _COHORT_KW):    return "cohort"
    return "execute_sql"


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
    graph.add_node("clean", clean_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("pivot", pivot_node)
    graph.add_node("stat_test", stat_test_node)
    graph.add_node("validate", validate_node)
    graph.add_node("cohort", cohort_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("visualize", visualize_node)
    graph.add_node("insights", insights_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("plan")

    # Route after plan to the right tool
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "clean": "clean",
            "forecast": "forecast",
            "pivot": "pivot",
            "stat_test": "stat_test",
            "validate": "validate",
            "cohort": "cohort",
            "execute_sql": "execute_sql",
        },
    )

    # All direct tool nodes skip retry and go straight to analyze
    for direct_node in ("clean", "forecast", "pivot", "stat_test", "validate", "cohort"):
        graph.add_edge(direct_node, "analyze")

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


def run_query(user_query: str, schema_text: str = "") -> dict[str, Any]:
    """Public entry point — invoke the agent and return the report."""
    initial_state: AgentState = {
        "query": user_query,
        "schema_text": schema_text,
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
