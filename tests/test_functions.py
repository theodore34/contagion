"""
Unit tests for functions.py

Run with:
    pytest tests/test_functions.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import load_data, correlation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(prices: dict, dates=None) -> pd.DataFrame:
    """Build a DataFrame with a 'date' column and the given price series."""
    if dates is None:
        n = len(next(iter(prices.values())))
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, **prices})


def _mock_read_csv(dfs_by_asset: dict):
    """Return a side_effect for pd.read_csv that serves per-asset DataFrames."""

    def _read(path, **kwargs):
        # path looks like "data/{asset}_filled.csv"
        asset = os.path.basename(path).replace("_filled.csv", "")
        return dfs_by_asset[asset].copy()

    return _read


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:

    def test_returns_dataframe(self):
        df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"])
        assert isinstance(result, pd.DataFrame)

    def test_index_is_date(self):
        df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"])
        assert result.index.name == "date"

    def test_log_returns_drops_first_row(self):
        """After shift(1) + dropna, exactly one row is removed."""
        n = 5
        df_a = _make_df({"price": [1.0] * n})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"])
        assert len(result) == n - 1

    def test_log_returns_values_doubling_series(self):
        """log(2^t / 2^{t-1}) = log(2) for every step of a doubling series."""
        df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"])
        np.testing.assert_allclose(result["price"].values, np.log(2.0))

    def test_no_log_returns_preserves_raw_prices(self):
        prices = [10.0, 20.0, 15.0, 25.0, 30.0]
        df_a = _make_df({"price": prices})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"], log_returns=False)
        np.testing.assert_array_equal(result["price"].values, prices)

    def test_no_log_returns_keeps_all_rows(self):
        prices = list(range(1, 7))
        df_a = _make_df({"price": prices})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"], log_returns=False)
        assert len(result) == len(prices)

    def test_merge_two_assets_contains_both_columns(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df_a = _make_df({"price_a": [1.0, 2.0, 3.0, 4.0, 5.0]}, dates=dates)
        df_b = _make_df({"price_b": [5.0, 4.0, 3.0, 2.0, 1.0]}, dates=dates)
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a, "B": df_b})):
            result = load_data(["A", "B"], log_returns=False)
        assert "price_a" in result.columns
        assert "price_b" in result.columns

    def test_inner_join_drops_non_overlapping_dates(self):
        """Only the 3 common dates (Jan 03-05) should be kept."""
        dates_a = pd.date_range("2023-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2023-01-03", periods=5, freq="D")
        df_a = _make_df({"price_a": [1.0, 2.0, 3.0, 4.0, 5.0]}, dates=dates_a)
        df_b = _make_df({"price_b": [5.0, 4.0, 3.0, 2.0, 1.0]}, dates=dates_b)
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a, "B": df_b})):
            result = load_data(["A", "B"], log_returns=False)
        assert len(result) == 3

    def test_log_returns_no_nan_in_result(self):
        df_a = _make_df({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
            result = load_data(["A"])
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

class TestCorrelation:

    def test_output_shape_no_lag(self):
        data = np.random.randn(100, 4)
        C = correlation(data, lag=0)
        assert C.shape == (4, 4)

    def test_output_shape_with_lag(self):
        data = np.random.randn(100, 4)
        C = correlation(data, lag=5)
        assert C.shape == (4, 4)

    def test_diagonal_ones_no_lag(self):
        """Self-correlation of any series must be 1."""
        data = np.random.randn(100, 3)
        C = correlation(data, lag=0)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)

    def test_symmetry_no_lag(self):
        data = np.random.randn(100, 3)
        C = correlation(data, lag=0)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_values_bounded_no_lag(self):
        data = np.random.randn(100, 5)
        C = correlation(data, lag=0)
        assert np.all(C >= -1.0 - 1e-12)
        assert np.all(C <= 1.0 + 1e-12)

    def test_matches_numpy_corrcoef_no_lag(self):
        data = np.random.randn(50, 3)
        C = correlation(data, lag=0)
        expected = np.corrcoef(data.T)
        np.testing.assert_allclose(C, expected, atol=1e-12)

    def test_perfect_correlation_identical_series_no_lag(self):
        """Two identical columns must yield an all-ones correlation matrix."""
        x = np.random.randn(80)
        data = np.column_stack([x, x])
        C = correlation(data, lag=0)
        np.testing.assert_allclose(C, np.ones((2, 2)), atol=1e-10)

    def test_lag_cross_correlation(self):
        """
        Construct data so that data[:-lag, 0] == data[lag:, 1].
        The lagged cross-correlation C[0, 1] must therefore be 1.0.

        C[0, 1] = corr(data[:-lag, 0], data[lag:, 1])
        """
        lag = 3
        rng = np.random.default_rng(42)
        x = rng.standard_normal(60)

        # Construct data so that data[:-lag, 0] and data[lag:, 1] are identical
        data = np.zeros((60, 2))
        data[:, 0] = x  # col 0 = x (reference series)
        data[lag:, 1] = x[:-lag]  # col 1[lag:] = x[:-lag]
        # Therefore:
        #   data[:-lag, 0] = x[:-lag]
        #   data[lag:, 1]  = x[:-lag]  ← identical series
        # So C[0, 1] should be 1.0

        C = correlation(data, lag=lag)

        np.testing.assert_allclose(
            C[0, 1], 1.0, atol=1e-10,
            err_msg=(
                "lag > 0 must return the cross-correlation block [:n, n:], "
                "not the auto-correlation block [:n, :n]"
            ),
        )
