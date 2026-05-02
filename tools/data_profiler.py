import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a full statistical profile of a DataFrame."""
    if df.empty:
        return {"error": "DataFrame is empty."}

    profile: dict[str, Any] = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "missing_summary": {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique())

        col_info: dict[str, Any] = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
            "unique_count": unique_count,
            "unique_pct": round(unique_count / len(df) * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update(
                {
                    "min": _safe_float(desc.get("min")),
                    "max": _safe_float(desc.get("max")),
                    "mean": _safe_float(desc.get("mean")),
                    "median": _safe_float(series.median()),
                    "std": _safe_float(desc.get("std")),
                    "skewness": _safe_float(series.skew()),
                    "q25": _safe_float(desc.get("25%")),
                    "q75": _safe_float(desc.get("75%")),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info.update(
                {
                    "min_date": str(series.min()),
                    "max_date": str(series.max()),
                    "date_range_days": (series.max() - series.min()).days
                    if not series.isna().all()
                    else None,
                }
            )
        else:
            top_values = series.value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    missing = {
        col: info["null_count"]
        for col, info in profile["columns"].items()
        if info["null_count"] > 0
    }
    profile["missing_summary"] = missing

    return profile


def profile_to_text(profile: dict[str, Any]) -> str:
    """Convert a profile dict to a human-readable string for LLM prompts."""
    if "error" in profile:
        return profile["error"]

    lines = [
        f"Dataset: {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns",
        f"Duplicate rows: {profile['duplicate_rows']:,}",
        "",
        "Column Details:",
    ]
    for col, info in profile["columns"].items():
        null_note = f" | {info['null_pct']}% nulls" if info["null_count"] > 0 else ""
        if "mean" in info:
            lines.append(
                f"  {col} ({info['dtype']}){null_note}: "
                f"min={info['min']}, max={info['max']}, "
                f"mean={info['mean']}, std={info['std']}"
            )
        elif "top_values" in info:
            top = ", ".join(f"{k}({v})" for k, v in list(info["top_values"].items())[:3])
            lines.append(
                f"  {col} ({info['dtype']}){null_note}: {info['unique_count']} unique — top: {top}"
            )
        else:
            lines.append(f"  {col} ({info['dtype']}){null_note}")

    return "\n".join(lines)


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return round(f, 4) if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None
