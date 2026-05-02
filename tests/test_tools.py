"""Unit tests for analytics tools — no LLM calls, no external dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from tools.data_profiler import profile_dataframe, profile_to_text
from tools.anomaly_detector import detect_anomalies
from tools.trend_detector import detect_trends
from tools.correlation_analyzer import analyze_correlations
from tools.sql_executor import QueryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "revenue": rng.normal(10000, 1500, 120),
            "quantity": rng.integers(10, 200, 120).astype(float),
            "region": rng.choice(["North", "South", "East"], 120),
            "category": rng.choice(["A", "B", "C"], 120),
        }
    )


@pytest.fixture
def df_with_anomalies(sample_df) -> pd.DataFrame:
    df = sample_df.copy()
    df.loc[5, "revenue"] = 150000   # extreme high
    df.loc[10, "revenue"] = -500    # extreme low
    df.loc[15, "quantity"] = 9999
    return df


# ---------------------------------------------------------------------------
# data_profiler
# ---------------------------------------------------------------------------

class TestDataProfiler:
    def test_shape(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert profile["shape"]["rows"] == 120
        assert profile["shape"]["columns"] == 5

    def test_numeric_stats_present(self, sample_df):
        profile = profile_dataframe(sample_df)
        rev = profile["columns"]["revenue"]
        assert "mean" in rev
        assert "std" in rev
        assert rev["null_count"] == 0

    def test_categorical_top_values(self, sample_df):
        profile = profile_dataframe(sample_df)
        assert "top_values" in profile["columns"]["region"]

    def test_empty_dataframe(self):
        result = profile_dataframe(pd.DataFrame())
        assert "error" in result

    def test_profile_to_text_returns_string(self, sample_df):
        profile = profile_dataframe(sample_df)
        text = profile_to_text(profile)
        assert isinstance(text, str)
        assert "revenue" in text


# ---------------------------------------------------------------------------
# anomaly_detector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def test_detects_injected_anomalies(self, df_with_anomalies):
        result = detect_anomalies(df_with_anomalies, method="both")
        assert result["total_anomalies"] > 0
        assert "revenue" in result["columns"] or "quantity" in result["columns"]

    def test_clean_data_low_anomalies(self, sample_df):
        result = detect_anomalies(sample_df, method="zscore", zscore_threshold=5.0)
        assert result["total_anomalies"] < 10

    def test_iqr_method(self, df_with_anomalies):
        result = detect_anomalies(df_with_anomalies, method="iqr")
        assert "iqr_outliers" in list(result["columns"].values())[0]

    def test_returns_summary_string(self, df_with_anomalies):
        result = detect_anomalies(df_with_anomalies)
        assert isinstance(result["summary"], str)

    def test_non_numeric_only_df(self):
        df = pd.DataFrame({"name": ["a", "b"], "city": ["x", "y"]})
        result = detect_anomalies(df)
        assert "error" in result


# ---------------------------------------------------------------------------
# trend_detector
# ---------------------------------------------------------------------------

class TestTrendDetector:
    def test_upward_trend(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": range(60)})
        result = detect_trends(df, date_col="date", value_col="revenue")
        assert result["direction"] == "upward"

    def test_downward_trend(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "revenue": range(60, 0, -1)})
        result = detect_trends(df, date_col="date", value_col="revenue")
        assert result["direction"] == "downward"

    def test_insufficient_data(self):
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "revenue": [100, 200]})
        result = detect_trends(df, date_col="date", value_col="revenue")
        # Should not error, just return minimal result
        assert "direction" in result or "error" in result


# ---------------------------------------------------------------------------
# correlation_analyzer
# ---------------------------------------------------------------------------

class TestCorrelationAnalyzer:
    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        result = analyze_correlations(df)
        top = result["top_pairs"][0]
        assert abs(top["correlation"] - 1.0) < 0.001

    def test_no_correlation(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
        result = analyze_correlations(df)
        assert abs(result["top_pairs"][0]["correlation"]) < 0.4

    def test_single_column_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = analyze_correlations(df)
        assert "error" in result


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------

class TestQueryResult:
    def test_success_to_dict(self, sample_df):
        qr = QueryResult(success=True, sql="SELECT 1", data=sample_df)
        d = qr.to_dict()
        assert d["success"] is True
        assert d["row_count"] == 120

    def test_failure_to_dict(self):
        qr = QueryResult(success=False, sql="BAD SQL", error="syntax error")
        d = qr.to_dict()
        assert d["success"] is False
        assert "error" in d
