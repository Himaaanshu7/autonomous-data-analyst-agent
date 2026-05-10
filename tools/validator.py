"""Data validation — check business rules and flag violations."""
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Built-in auto-rules applied to every dataset
_AUTO_RULES = [
    {"name": "No negative values", "type": "auto_negative"},
    {"name": "No future dates", "type": "auto_future_date"},
    {"name": "Duplicate rows", "type": "auto_duplicates"},
    {"name": "High null rate (>30%)", "type": "auto_nulls"},
]


def validate(
    df: pd.DataFrame,
    custom_rules: list[dict] | None = None,
) -> dict[str, Any]:
    """Run auto + custom validation rules. Returns structured report."""
    results: list[dict] = []
    total_issues = 0

    # ── Auto rules ──────────────────────────────────────────────────────
    # 1. Negative values in numeric columns
    for col in df.select_dtypes(include="number").columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            results.append({
                "rule": f"No negative values in '{col}'",
                "status": "FAIL",
                "violations": int(neg_count),
                "pct": round(neg_count / len(df) * 100, 1),
                "sample_rows": df[df[col] < 0].head(3).to_dict(orient="records"),
            })
            total_issues += neg_count

    # 2. Future dates
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                future = (parsed > pd.Timestamp.now()).sum()
                if future > 0:
                    results.append({
                        "rule": f"No future dates in '{col}'",
                        "status": "FAIL",
                        "violations": int(future),
                        "pct": round(future / len(df) * 100, 1),
                        "sample_rows": [],
                    })
                    total_issues += future
            except Exception:
                pass

    # 3. Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        results.append({
            "rule": "No duplicate rows",
            "status": "FAIL",
            "violations": int(dup_count),
            "pct": round(dup_count / len(df) * 100, 1),
            "sample_rows": df[df.duplicated()].head(3).to_dict(orient="records"),
        })
        total_issues += dup_count

    # 4. High null rate
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct > 30:
            results.append({
                "rule": f"'{col}' null rate > 30%",
                "status": "WARN",
                "violations": int(df[col].isna().sum()),
                "pct": round(null_pct, 1),
                "sample_rows": [],
            })

    # ── Custom rules ────────────────────────────────────────────────────
    for rule in (custom_rules or []):
        col = rule.get("column")
        op = rule.get("operator")
        val = rule.get("value")
        label = rule.get("name", f"{col} {op} {val}")
        if col not in df.columns:
            continue
        try:
            mask = _apply_rule(df[col], op, val)
            viol = (~mask).sum()
            results.append({
                "rule": label,
                "status": "FAIL" if viol > 0 else "PASS",
                "violations": int(viol),
                "pct": round(viol / len(df) * 100, 1),
                "sample_rows": df[~mask].head(3).to_dict(orient="records"),
            })
            total_issues += viol
        except Exception as exc:
            logger.warning("Custom rule failed: %s", exc)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    warned = [r for r in results if r["status"] == "WARN"]

    quality_score = max(0, 100 - (total_issues / max(len(df), 1) * 100))

    return {
        "total_rows": len(df),
        "total_issues": int(total_issues),
        "quality_score": round(quality_score, 1),
        "passed": len(passed),
        "failed": len(failed),
        "warnings": len(warned),
        "results": results,
        "summary": (
            f"Data quality score: {quality_score:.0f}/100. "
            f"{len(failed)} rule(s) failed, {len(warned)} warning(s). "
            f"{total_issues:,} total violations across {len(df):,} rows."
        ),
    }


def _apply_rule(series: pd.Series, op: str, val: Any) -> pd.Series:
    ops = {">": series > val, ">=": series >= val, "<": series < val,
           "<=": series <= val, "==": series == val, "!=": series != val,
           "not_null": series.notna(), "unique": ~series.duplicated()}
    return ops.get(op, pd.Series([True] * len(series)))
