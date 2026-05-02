import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import settings

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "both",
    zscore_threshold: float | None = None,
    iqr_multiplier: float | None = None,
) -> dict[str, Any]:
    """Detect anomalies in all numeric columns using IQR and/or Z-score.

    Returns a structured dict with per-column findings and flagged rows.
    method: "iqr" | "zscore" | "both"
    """
    z_thresh = zscore_threshold or settings.anomaly_zscore_threshold
    iqr_mult = iqr_multiplier or settings.anomaly_iqr_multiplier

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns found.", "columns": {}, "total_anomalies": 0}

    results: dict[str, Any] = {
        "method": method,
        "columns": {},
        "anomaly_rows": [],
        "total_anomalies": 0,
        "summary": "",
    }

    all_anomaly_indices: set[int] = set()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        col_result: dict[str, Any] = {"anomaly_indices": [], "bounds": {}}

        if method in ("iqr", "both"):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
            iqr_outliers = df.index[
                (df[col] < lower) | (df[col] > upper)
            ].tolist()
            col_result["iqr_outliers"] = len(iqr_outliers)
            col_result["bounds"]["iqr"] = {
                "lower": round(float(lower), 4),
                "upper": round(float(upper), 4),
            }
            col_result["anomaly_indices"].extend(iqr_outliers)

        if method in ("zscore", "both"):
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            z_series = pd.Series(z_scores, index=series.index)
            z_outliers = df.index[z_series > z_thresh].tolist()
            col_result["zscore_outliers"] = len(z_outliers)
            col_result["bounds"]["zscore_threshold"] = z_thresh
            col_result["anomaly_indices"].extend(z_outliers)

        # Deduplicate
        col_result["anomaly_indices"] = list(set(col_result["anomaly_indices"]))
        col_result["total"] = len(col_result["anomaly_indices"])
        all_anomaly_indices.update(col_result["anomaly_indices"])

        if col_result["total"] > 0:
            results["columns"][col] = col_result

    # Collect flagged rows (up to 50 for readability)
    if all_anomaly_indices:
        anomaly_df = df.loc[sorted(all_anomaly_indices)[:50]]
        results["anomaly_rows"] = anomaly_df.to_dict(orient="records")

    results["total_anomalies"] = len(all_anomaly_indices)
    results["summary"] = _build_summary(results["columns"], len(df))
    return results


def _build_summary(col_results: dict, total_rows: int) -> str:
    if not col_results:
        return "No anomalies detected."
    lines = []
    for col, info in col_results.items():
        pct = round(info["total"] / total_rows * 100, 1)
        lines.append(f"  {col}: {info['total']} anomalies ({pct}% of rows)")
    return "Anomalies found:\n" + "\n".join(lines)
