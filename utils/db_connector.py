import logging
from pathlib import Path

import duckdb
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

_connection: duckdb.DuckDBPyConnection | None = None


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
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0]
    return count > 0


def load_csv_as_table(csv_path: str, table_name: str, overwrite: bool = False) -> None:
    conn = get_connection()
    if table_exists(table_name) and not overwrite:
        logger.info("Table '%s' already loaded — skipping.", table_name)
        return
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(
        f'CREATE TABLE "{table_name}" AS SELECT * FROM read_csv_auto(\'{csv_path}\', header=true)'
    )
    count = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
    logger.info("Loaded '%s' — %s rows.", table_name, f"{count:,}")


def load_all_sample_data(overwrite: bool = False) -> list[str]:
    sample_dir = Path(settings.sample_data_dir)
    loaded: list[str] = []
    for csv_path in sorted(sample_dir.glob("*.csv")):
        table_name = csv_path.stem.lower().replace("-", "_").replace(" ", "_")
        load_csv_as_table(str(csv_path), table_name, overwrite=overwrite)
        loaded.append(table_name)
    return loaded


def list_tables() -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def run_query(sql: str) -> pd.DataFrame:
    return get_connection().execute(sql).df()


def close() -> None:
    global _connection
    if _connection:
        _connection.close()
        _connection = None
