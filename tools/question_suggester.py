"""Auto-generate relevant questions based on dataset schema."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def suggest_questions(schema_text: str, n: int = 6) -> list[str]:
    """Use the LLM to suggest dataset-specific questions."""
    from utils.llm_client import llm_client

    system = (
        "You are a data analyst. Given a database schema, suggest practical business questions "
        "a user might want to ask. Return ONLY a JSON array of question strings, nothing else.\n"
        "Example: [\"What is the total revenue by region?\", \"Show top 5 customers by value\"]"
    )
    user = (
        f"Schema:\n{schema_text[:2000]}\n\n"
        f"Suggest {n} specific, practical questions the user can ask about this data. "
        f"Use actual column names from the schema. Return JSON array only."
    )

    try:
        result = llm_client.complete_json(system=system, user=user)
        if isinstance(result, list):
            return [str(q) for q in result[:n]]
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return [str(q) for q in v[:n]]
    except Exception as exc:
        logger.warning("Question suggester failed: %s", exc)

    # Fallback generic questions
    return [
        "Show me the first 10 rows of this dataset",
        "Give me a full profile of this data",
        "Detect any anomalies in this dataset",
        "What are the top 5 records by the main numeric column?",
        "Show the distribution of values in each column",
        "Are there any missing values I should know about?",
    ]
