import re
import logging

from agents.prompts import (
    SQL_SYSTEM_PROMPT,
    SQL_REFLECTION_PROMPT,
    build_sql_user_prompt,
)
from utils.llm_client import llm_client
from utils.schema_inspector import get_all_schemas_as_text

logger = logging.getLogger(__name__)

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_sql(text: str) -> str:
    match = _SQL_BLOCK.search(text)
    return match.group(1).strip() if match else text.strip()


def generate_sql(natural_language_query: str, schema_text: str | None = None) -> str:
    """Convert a natural language query to DuckDB SQL.

    schema_text is injected into the cached system prompt so repeated calls
    with the same schema hit the Anthropic prompt cache.
    """
    if schema_text is None:
        schema_text = get_all_schemas_as_text()

    system = SQL_SYSTEM_PROMPT.format(schema=schema_text)
    user = build_sql_user_prompt(natural_language_query)

    response = llm_client.complete(system=system, user=user, cache_system=True)
    sql = _extract_sql(response)
    logger.debug("Generated SQL:\n%s", sql)
    return sql


def fix_sql(
    user_query: str,
    failed_sql: str,
    error_message: str,
    schema_text: str | None = None,
) -> str:
    """Ask the model to repair a failing SQL query."""
    if schema_text is None:
        schema_text = get_all_schemas_as_text()

    prompt = SQL_REFLECTION_PROMPT.format(
        user_query=user_query,
        failed_sql=failed_sql,
        error_message=error_message,
        schema=schema_text,
    )
    response = llm_client.complete(system="You are a DuckDB SQL expert.", user=prompt)
    return _extract_sql(response)
