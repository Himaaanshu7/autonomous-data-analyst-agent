import logging
from typing import Any

import pandas as pd

from utils.db_connector import get_connection, list_tables

logger = logging.getLogger(__name__)


def get_table_schema(table_name: str) -> dict[str, Any]:
    conn = get_connection()
    cols_df: pd.DataFrame = conn.execute(f'PRAGMA table_info("{table_name}")').df()
    sample_df: pd.DataFrame = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3').df()
    row_count: int = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]

    return {
        "table": table_name,
        "row_count": row_count,
        "columns": [
            {"name": row["name"], "type": row["type"]}
            for _, row in cols_df.iterrows()
        ],
        "sample_rows": sample_df.to_dict(orient="records"),
    }


def get_all_schemas_as_text() -> str:
    tables = list_tables()
    if not tables:
        return "No tables loaded. Please load a dataset first."

    parts: list[str] = []
    for table in tables:
        try:
            schema = get_table_schema(table)
            cols_text = ", ".join(
                f"{c['name']} ({c['type']})" for c in schema["columns"]
            )
            sample_lines = "\n  ".join(str(r) for r in schema["sample_rows"][:2])
            parts.append(
                f"TABLE: {table}  ({schema['row_count']:,} rows)\n"
                f"COLUMNS: {cols_text}\n"
                f"SAMPLE:\n  {sample_lines}"
            )
        except Exception as exc:
            logger.warning("Could not inspect table '%s': %s", table, exc)

    return "\n\n".join(parts)
