from typing import Annotated, Any
import operator
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    query: str
    schema_text: str

    # Planning
    plan: dict[str, Any]

    # SQL execution
    sql: str
    sql_result: dict[str, Any]      # {"success": bool, "data": [...], "columns": [...], "error": str}

    # Analysis outputs
    profile: dict[str, Any]
    anomalies: dict[str, Any]
    trends: dict[str, Any]
    correlations: dict[str, Any]

    # Presentation
    visualizations: list[dict]      # list of Plotly figure dicts (serialisable)
    insights: dict[str, Any]
    report: dict[str, Any]

    # Control flow
    error: str
    retry_count: int
    messages: Annotated[list[dict], operator.add]
