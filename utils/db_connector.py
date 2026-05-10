import logging
from pathlib import Path

import duckdb
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

_connection: duckdb.DuckDBPyConnection | None = None


def _sanitize_col(name: str) -> str:
    """Convert any column name to safe snake_case for SQL."""
    import re
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)  # replace spaces/special chars with _
    name = re.sub(r"_+", "_", name).strip("_")  # collapse multiple underscores
    if name and name[0].isdigit():
        name = "col_" + name               # prevent names starting with a digit
    return name or "unnamed"


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        db_path = Path(settings.duckdb_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _connection = duckdb.connect(str(db_path))
        logger.info("DuckDB connected at %s", db_path)
    return _connection


def table_exists(table_name: str) -> bool:
    conn = get_connection()
    try:
        result = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result is not None and result[0] > 0
    except Exception:
        return False


def load_csv_as_table(csv_path: str, table_name: str, overwrite: bool = False) -> None:
    conn = get_connection()
    if table_exists(table_name) and not overwrite:
        logger.info("Table '%s' already loaded — skipping.", table_name)
        return

    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    # Always load via pandas so we can sanitize column names before inserting.
    # Column names with spaces break SQL generation — snake_case is safer.
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            df.columns = [_sanitize_col(c) for c in df.columns]
            conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
            count = len(df)
            logger.info("Loaded '%s' via pandas (%s) — %s rows.", table_name, encoding, f"{count:,}")
            return
        except Exception as e2:
            logger.warning("Pandas load with %s failed: %s", encoding, e2)

    raise RuntimeError(
        f"Could not load '{Path(csv_path).name}'. "
        "Make sure it is a valid CSV file with a header row."
    )


def load_all_sample_data(overwrite: bool = False) -> list[str]:
    sample_dir = Path(settings.sample_data_dir)
    loaded: list[str] = []
    for csv_path in sorted(sample_dir.glob("*.csv")):
        table_name = csv_path.stem.lower().replace("-", "_").replace(" ", "_")
        load_csv_as_table(str(csv_path), table_name, overwrite=overwrite)
        loaded.append(table_name)
    return loaded


def drop_table(table_name: str) -> None:
    conn = get_connection()
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    logger.info("Dropped table '%s'.", table_name)


def list_tables() -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def run_query(sql: str) -> pd.DataFrame:
    return get_connection().execute(sql).df()


def execute_statement(sql: str) -> None:
    """Execute a DDL/DML statement that returns no result (ALTER, UPDATE, CREATE VIEW, etc.)."""
    get_connection().execute(sql)
    logger.info("Executed statement: %s", sql[:120])


def create_filtered_view(table: str, where_clause: str) -> str:
    """Create a DuckDB view filtering `table` by `where_clause`. Returns view name."""
    view_name = f"{table}_filtered"
    sql = f'CREATE OR REPLACE VIEW "{view_name}" AS SELECT * FROM "{table}" WHERE {where_clause}'
    execute_statement(sql)
    return view_name


def drop_view(view_name: str) -> None:
    execute_statement(f'DROP VIEW IF EXISTS "{view_name}"')


def list_views() -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'VIEW' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def close() -> None:
    global _connection
    if _connection:
        _connection.close()
        _connection = None
