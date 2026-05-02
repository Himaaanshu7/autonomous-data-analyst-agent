import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute correlation matrix and extract top correlated pairs."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis."}

    corr_matrix = numeric_df.corr(method=method)

    # Extract upper triangle pairs
    pairs: list[dict] = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                pairs.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "correlation": round(float(val), 4),
                        "strength": _label(abs(val)),
                        "direction": "positive" if val > 0 else "negative",
                    }
                )

    pairs_sorted = sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)

    return {
        "method": method,
        "matrix": {
            col: {
                c: (round(float(v), 4) if not np.isnan(v) else None)
                for c, v in corr_matrix[col].items()
            }
            for col in corr_matrix.columns
        },
        "top_pairs": pairs_sorted[:top_n],
        "strong_pairs": [p for p in pairs_sorted if abs(p["correlation"]) >= 0.7],
        "columns_analyzed": list(numeric_df.columns),
        "summary": _build_summary(pairs_sorted[:3]),
    }


def _label(r: float) -> str:
    if r >= 0.9:
        return "very strong"
    if r >= 0.7:
        return "strong"
    if r >= 0.5:
        return "moderate"
    if r >= 0.3:
        return "weak"
    return "negligible"


def _build_summary(top_pairs: list[dict]) -> str:
    if not top_pairs:
        return "No notable correlations found."
    lines = []
    for p in top_pairs:
        lines.append(
            f"{p['col_a']} ↔ {p['col_b']}: {p['correlation']:+.3f} ({p['strength']} {p['direction']})"
        )
    return "Top correlations: " + "; ".join(lines)
