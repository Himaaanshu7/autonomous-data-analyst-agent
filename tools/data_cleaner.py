"""Data cleaning tool — detects and fixes common data quality issues.

Supported operations:
  - Drop duplicate rows
  - Drop columns that are entirely null
  - Fill missing values (numeric → median, categorical → mode)
  - Strip whitespace from string columns
  - Standardize column names to snake_case
  - Fix mixed-type columns (cast to best inferred type)
  - Remove rows where key columns are null
  - Cap outliers at IQR bounds (optional)

Returns the cleaned DataFrame and a detailed cleaning report dict.
"""
import re
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_missing: bool = True,
    strip_strings: bool = True,
    fix_dtypes: bool = True,
    drop_all_null_cols: bool = True,
    cap_outliers: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a DataFrame and return (cleaned_df, report).

    The report documents every change made so it can be shown to the user.
    """
    original_shape = df.shape
    df = df.copy()
    report: dict[str, Any] = {
        "original_rows": original_shape[0],
        "original_cols": original_shape[1],
        "steps": [],
        "columns_changed": [],
    }

    # ── 1. Standardise column names ──────────────────────────────────────
    old_cols = list(df.columns)
    df.columns = [_snake(c) for c in df.columns]
    renamed = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]
    if renamed:
        report["steps"].append({
            "operation": "rename_columns",
            "detail": f"Renamed {len(renamed)} columns to snake_case",
            "examples": [f"'{o}' → '{n}'" for o, n in renamed[:5]],
        })

    # ── 2. Strip whitespace from all string columns ───────────────────────
    if strip_strings:
        stripped = []
        for col in df.select_dtypes(include="object").columns:
            before = df[col].isna().sum()
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
            after = df[col].isna().sum()
            stripped.append(col)
        if stripped:
            report["steps"].append({
                "operation": "strip_whitespace",
                "detail": f"Stripped whitespace from {len(stripped)} text columns",
                "columns": stripped,
            })

    # ── 3. Drop fully-null columns ────────────────────────────────────────
    if drop_all_null_cols:
        null_cols = [c for c in df.columns if df[c].isna().all()]
        if null_cols:
            df.drop(columns=null_cols, inplace=True)
            report["steps"].append({
                "operation": "drop_null_columns",
                "detail": f"Dropped {len(null_cols)} fully-null columns",
                "columns": null_cols,
            })

    # ── 4. Fix inferred data types ────────────────────────────────────────
    if fix_dtypes:
        type_fixes = []
        for col in df.select_dtypes(include="object").columns:
            # Try numeric
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() / max(df[col].notna().sum(), 1) > 0.9:
                df[col] = numeric
                type_fixes.append(f"'{col}' → numeric")
                continue
            # Try datetime
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                    df[col] = parsed
                    type_fixes.append(f"'{col}' → datetime")
            except Exception:
                pass
        if type_fixes:
            report["steps"].append({
                "operation": "fix_dtypes",
                "detail": f"Fixed data types for {len(type_fixes)} columns",
                "examples": type_fixes[:5],
            })

    # ── 5. Drop duplicate rows ────────────────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df.drop_duplicates(inplace=True)
        removed = before - len(df)
        if removed > 0:
            report["steps"].append({
                "operation": "drop_duplicates",
                "detail": f"Removed {removed:,} duplicate rows",
                "rows_removed": removed,
            })

    # ── 6. Fill missing values ────────────────────────────────────────────
    if fill_missing:
        fill_log = []
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            pct = null_count / len(df) * 100

            if pct > 60:
                # Too many nulls — drop the column
                df.drop(columns=[col], inplace=True)
                fill_log.append(f"'{col}': dropped ({pct:.0f}% null)")
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                fill_log.append(f"'{col}': filled {null_count} nulls with median ({fill_val:.2f})")
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col].fillna(mode_vals[0], inplace=True)
                    fill_log.append(f"'{col}': filled {null_count} nulls with mode ('{mode_vals[0]}')")

        if fill_log:
            report["steps"].append({
                "operation": "fill_missing",
                "detail": f"Handled missing values in {len(fill_log)} columns",
                "log": fill_log,
            })

    # ── 7. Cap outliers at IQR bounds (optional) ─────────────────────────
    if cap_outliers:
        capped = []
        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            before = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if before > 0:
                capped.append(f"'{col}': capped {before} outliers [{lower:.2f}, {upper:.2f}]")
        if capped:
            report["steps"].append({
                "operation": "cap_outliers",
                "detail": f"Capped outliers in {len(capped)} columns",
                "log": capped,
            })

    # ── Final summary ─────────────────────────────────────────────────────
    report["final_rows"] = len(df)
    report["final_cols"] = len(df.columns)
    report["rows_removed"] = original_shape[0] - len(df)
    report["cols_removed"] = original_shape[1] - len(df.columns)
    report["total_steps"] = len(report["steps"])
    report["summary"] = (
        f"Cleaned dataset: {original_shape[0]:,} rows × {original_shape[1]} cols "
        f"→ {len(df):,} rows × {len(df.columns)} cols. "
        f"{report['rows_removed']:,} rows removed, "
        f"{report['cols_removed']} columns removed/converted, "
        f"{len(report['steps'])} operations applied."
    )

    return df, report


def save_cleaned_table(df: pd.DataFrame, table_name: str, suffix: str = "_cleaned") -> str:
    """Save cleaned DataFrame back to DuckDB as a new table and return its name."""
    from utils.db_connector import get_connection
    conn = get_connection()
    clean_name = table_name.rstrip(suffix) + suffix
    conn.execute(f'DROP TABLE IF EXISTS "{clean_name}"')
    conn.execute(f'CREATE TABLE "{clean_name}" AS SELECT * FROM df')
    return clean_name


def _snake(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return ("col_" + name) if name and name[0].isdigit() else (name or "unnamed")
