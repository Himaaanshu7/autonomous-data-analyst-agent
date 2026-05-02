import logging
from dataclasses import dataclass, field

import pandas as pd

from utils.db_connector import run_query

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    success: bool
    sql: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ""

    @property
    def row_count(self) -> int:
        return len(self.data)

    def to_dict(self) -> dict:
        """Serialisable representation for agent state."""
        if not self.success:
            return {"success": False, "error": self.error, "sql": self.sql}
        return {
            "success": True,
            "sql": self.sql,
            "columns": list(self.data.columns),
            "row_count": self.row_count,
            "data": self.data.head(500).to_dict(orient="records"),
            "data_summary": self.data.describe(include="all").to_string(),
        }


def execute_sql(sql: str) -> QueryResult:
    """Execute SQL against DuckDB and return a QueryResult."""
    try:
        df = run_query(sql)
        logger.debug("Query returned %d rows.", len(df))
        return QueryResult(success=True, sql=sql, data=df)
    except Exception as exc:
        logger.warning("SQL execution failed: %s\nSQL: %s", exc, sql)
        return QueryResult(success=False, sql=sql, error=str(exc))
