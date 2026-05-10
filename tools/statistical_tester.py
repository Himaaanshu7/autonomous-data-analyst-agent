"""Statistical significance testing — t-test, chi-square, ANOVA."""
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def run_statistical_tests(
    df: pd.DataFrame,
    group_col: str | None = None,
    value_col: str | None = None,
) -> dict[str, Any]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    group_col = group_col or (cat_cols[0] if cat_cols else None)
    value_col = value_col or (num_cols[0] if num_cols else None)

    if not group_col or not value_col:
        return {"error": "Need a categorical column (group) and a numeric column (value)."}

    groups = df.groupby(group_col)[value_col].apply(list).to_dict()
    group_names = list(groups.keys())
    results: dict[str, Any] = {
        "group_col": group_col,
        "value_col": value_col,
        "groups": {},
        "tests": [],
        "summary": "",
    }

    # Descriptive stats per group
    for name, vals in groups.items():
        arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
        results["groups"][str(name)] = {
            "n": len(arr),
            "mean": round(float(arr.mean()), 4) if len(arr) else None,
            "std": round(float(arr.std()), 4) if len(arr) else None,
            "median": round(float(np.median(arr)), 4) if len(arr) else None,
        }

    clean_groups = [
        np.array([v for v in vals if v is not None and not np.isnan(float(v))])
        for vals in groups.values()
    ]
    clean_groups = [g for g in clean_groups if len(g) >= 2]

    if len(clean_groups) < 2:
        return {**results, "error": "Need at least 2 groups with sufficient data."}

    # T-test (2 groups) or ANOVA (3+)
    if len(clean_groups) == 2:
        stat, p = stats.ttest_ind(clean_groups[0], clean_groups[1], equal_var=False)
        results["tests"].append({
            "test": "Welch's t-test",
            "groups": [str(g) for g in group_names[:2]],
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "interpretation": (
                f"Significant difference between {group_names[0]} and {group_names[1]}."
                if p < 0.05 else
                f"No significant difference between {group_names[0]} and {group_names[1]}."
            ),
        })
    else:
        stat, p = stats.f_oneway(*clean_groups)
        results["tests"].append({
            "test": "One-way ANOVA",
            "groups": [str(g) for g in group_names],
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "interpretation": (
                "At least one group mean is significantly different."
                if p < 0.05 else
                "No significant difference between group means."
            ),
        })

    # Chi-square test on group sizes
    observed = [len(v) for v in groups.values()]
    expected = [sum(observed) / len(observed)] * len(observed)
    chi2, p_chi = stats.chisquare(observed, expected)
    results["tests"].append({
        "test": "Chi-square (group size balance)",
        "statistic": round(float(chi2), 4),
        "p_value": round(float(p_chi), 6),
        "significant": p_chi < 0.05,
        "interpretation": (
            "Groups are significantly unequal in size." if p_chi < 0.05
            else "Groups are roughly balanced in size."
        ),
    })

    significant_tests = [t for t in results["tests"] if t["significant"]]
    results["summary"] = (
        f"Tested '{value_col}' across {len(group_names)} groups of '{group_col}'. "
        + (f"{len(significant_tests)} significant result(s) found."
           if significant_tests else "No significant differences found.")
    )
    return results
