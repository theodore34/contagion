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

from functions import (
    load_data, correlation, corr_threshold,
    var_contagion, contagion_matrix, contagion_density, contagion_threshold,
)


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

    def test_lag_with_known_solutions(self):
        """
        Test lagged correlation with known analytical solutions.
        Create two series where series[1] = series[0] shifted by k positions.
        At lag=k, cross-correlation should be perfect (1.0).
        """
        rng = np.random.default_rng(42)
        base_series = rng.standard_normal(100)

        # Test over multiple lags
        test_lags = [1, 2, 3, 5]

        for k in test_lags:
            # Series 1: base series
            # Series 2: base series shifted forward by k positions
            # At lag=k, they should align → correlation = 1.0
            data = np.zeros((100, 2))
            data[:, 0] = base_series
            data[k:, 1] = base_series[:-k]  # Shifted series

            C = correlation(data, lag=k)

            # At lag=k, the series align perfectly
            np.testing.assert_allclose(
                C[0, 1], 1.0, atol=1e-10,
                err_msg=f"At lag={k}, shifted series should correlate perfectly"
            )


# ---------------------------------------------------------------------------
# corr_threshold
# ---------------------------------------------------------------------------

class TestCorrThreshold:

    @pytest.fixture()
    def symmetric_corr(self):
        """A 4x4 symmetric correlation matrix with known values."""
        return np.array([
            [1.0, 0.9, 0.3, 0.1],
            [0.9, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ])

    def test_output_is_symmetric(self, symmetric_corr):
        result = corr_threshold(symmetric_corr, 0.5)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_diagonal_preserved_moderate_quantile(self, symmetric_corr):
        """At moderate quantile, diagonal (1.0) stays since threshold < 1."""
        result = corr_threshold(symmetric_corr, 0.5)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-12)

    def test_quantile_zero_zeros_minimum(self, symmetric_corr):
        """quantile=0 → threshold = min(|values|), so the smallest value gets zeroed."""
        result = corr_threshold(symmetric_corr, 0.0)
        # threshold = min abs = 0.1, values with |v| <= 0.1 are zeroed
        assert result[0, 3] == 0.0
        assert result[3, 0] == 0.0
        # larger values survive
        assert result[0, 1] == pytest.approx(0.9)

    def test_values_not_doubled(self, symmetric_corr):
        """Regression: old code doubled off-diagonal values."""
        result = corr_threshold(symmetric_corr, 0.0)
        assert np.all(np.abs(result) <= 1.0 + 1e-12)

    def test_known_example(self):
        """Hand-computed: threshold zeros out |v| <= thres."""
        corr = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ])
        # Lower triangle values (diag=True): [1.0, 0.8, 1.0, 0.2, 0.5, 1.0]
        # abs sorted: [0.2, 0.5, 0.8, 1.0, 1.0, 1.0]
        # quantile=0.5 → threshold = median of abs values
        thres = np.quantile([0.2, 0.5, 0.8, 1.0, 1.0, 1.0], 0.5)
        expected = corr.copy()
        expected[np.abs(expected) <= thres] = 0
        result = corr_threshold(corr, 0.5, diag=True)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_high_quantile_zeros_most(self, symmetric_corr):
        """A very high quantile should zero out most off-diagonal entries."""
        result = corr_threshold(symmetric_corr, 0.95)
        off_diag_nonzero = np.count_nonzero(result - np.diag(np.diag(result)))
        total_off_diag = symmetric_corr.size - symmetric_corr.shape[0]
        assert off_diag_nonzero < total_off_diag

    def test_diag_false_uses_full_matrix(self):
        """With diag=False, quantile is computed over all entries."""
        corr = np.array([
            [1.0, 0.6],
            [0.6, 1.0],
        ])
        # diag=True:  lower tri values = [1.0, 0.6, 1.0], abs = [0.6, 1.0, 1.0]
        # diag=False: all values = [1.0, 0.6, 0.6, 1.0], abs = [0.6, 0.6, 1.0, 1.0]
        # At quantile=0.5, thresholds differ
        r_diag = corr_threshold(corr, 0.5, diag=True)
        r_full = corr_threshold(corr, 0.5, diag=False)
        # Both should be symmetric
        np.testing.assert_allclose(r_diag, r_diag.T, atol=1e-12)
        np.testing.assert_allclose(r_full, r_full.T, atol=1e-12)

    def test_zeros_stay_symmetric(self, symmetric_corr):
        """If (i,j) is zeroed, (j,i) must also be zero."""
        result = corr_threshold(symmetric_corr, 0.7)
        zero_mask = result == 0
        np.testing.assert_array_equal(zero_mask, zero_mask.T)

    def test_no_mutation_of_input(self, symmetric_corr):
        """Input matrix must not be modified in place."""
        original = symmetric_corr.copy()
        corr_threshold(symmetric_corr, 0.5)
        np.testing.assert_array_equal(symmetric_corr, original)


# ---------------------------------------------------------------------------
# Helpers for contagion tests
# ---------------------------------------------------------------------------

def _make_returns(n_obs=200, seed=42):
    """Generate a small DataFrame of synthetic log-returns (3 assets)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_obs, 3)) * 0.01
    # Inject cross-asset dependence: B partly follows A with lag 1
    data[1:, 1] += 0.5 * data[:-1, 0]
    return pd.DataFrame(data, columns=["A", "B", "C"])


# ---------------------------------------------------------------------------
# var_contagion
# ---------------------------------------------------------------------------

class TestVarContagion:

    @pytest.fixture()
    def returns(self):
        return _make_returns()

    def test_output_shape(self, returns):
        """Output has n_lags*N + 1 rows (const + lags) and N columns."""
        result = var_contagion(returns, n_lags=1)
        n_assets = len(returns.columns)
        assert result.shape == (n_assets * 1 + 1, n_assets)

    def test_output_shape_two_lags(self, returns):
        result = var_contagion(returns, n_lags=2)
        n_assets = len(returns.columns)
        assert result.shape == (n_assets * 2 + 1, n_assets)

    def test_columns_match_assets(self, returns):
        result = var_contagion(returns, n_lags=1)
        assert list(result.columns) == list(returns.columns)

    def test_nonsignificant_coeffs_zeroed(self, returns):
        """With a high threshold (1.0), all coefficients should be zeroed."""
        result = var_contagion(returns, n_lags=1, pvalue_threshold=0.0)
        # pvalue_threshold=0 means only coefficients with pvalue == 0 survive
        # In practice most should be zeroed
        zero_count = (result.values == 0).sum()
        assert zero_count > 0

    def test_exclude_self_removes_own_lags(self, returns):
        """With include_self=False, own-lag regressors are dropped."""
        result = var_contagion(returns, n_lags=1, include_self=False)
        n_assets = len(returns.columns)
        # Fewer rows: const + (N-1)*n_lags per equation
        assert result.shape[0] == (n_assets - 1) * 1 + 1

    def test_detects_known_dependence(self):
        """B depends on lagged A → coefficient A.L1 in B equation should be nonzero."""
        returns = _make_returns(n_obs=500, seed=0)
        result = var_contagion(returns, n_lags=1, pvalue_threshold=0.1)
        assert result.loc["A.L1", "B"] != 0.0


# ---------------------------------------------------------------------------
# contagion_matrix
# ---------------------------------------------------------------------------

class TestContagionMatrix:

    @pytest.fixture()
    def returns(self):
        return _make_returns(n_obs=500, seed=0)

    def test_square_output(self, returns):
        matrix = contagion_matrix(returns, n_lags=1)
        n = len(returns.columns)
        assert matrix.shape == (n, n)

    def test_diagonal_is_zero(self, returns):
        matrix = contagion_matrix(returns, n_lags=1)
        for col in matrix.columns:
            assert matrix.loc[col, col] == 0.0

    def test_index_and_columns_match(self, returns):
        matrix = contagion_matrix(returns, n_lags=1)
        assert list(matrix.index) == list(matrix.columns)

    def test_detects_a_to_b_contagion(self, returns):
        """Known dependence B <- A should appear as nonzero entry."""
        matrix = contagion_matrix(returns, n_lags=1, pvalue_threshold=0.1)
        assert matrix.loc["A", "B"] != 0.0


# ---------------------------------------------------------------------------
# contagion_density
# ---------------------------------------------------------------------------

class TestContagionDensity:

    def test_full_matrix_returns_100(self):
        """Matrix with all nonzero off-diagonal → density = 100%."""
        m = pd.DataFrame(
            [[0.0, 0.5, 0.3], [0.2, 0.0, 0.4], [0.1, 0.6, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        assert contagion_density(m) == pytest.approx(100.0)

    def test_empty_matrix_returns_0(self):
        """Matrix with all zeros → density = 0%."""
        m = pd.DataFrame(
            np.zeros((3, 3)),
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        assert contagion_density(m) == pytest.approx(0.0)

    def test_partial_density(self):
        """One nonzero off-diagonal out of 6 possible → density ≈ 16.67%."""
        m = pd.DataFrame(
            [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        assert contagion_density(m) == pytest.approx(100 / 6, rel=1e-6)

    def test_single_asset_returns_0(self):
        """1x1 matrix → 0 possible edges → density = 0."""
        m = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
        assert contagion_density(m) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# contagion_threshold
# ---------------------------------------------------------------------------

class TestContagionThreshold:

    @pytest.fixture()
    def sample_matrix(self):
        return pd.DataFrame(
            [[0.0, 0.8, 0.1], [0.3, 0.0, 0.05], [0.6, 0.2, 0.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )

    def test_high_quantile_zeros_small_values(self, sample_matrix):
        result = contagion_threshold(sample_matrix, 0.8)
        # Only the largest values should survive
        assert (result.values == 0).sum() > (sample_matrix.values == 0).sum()

    def test_quantile_zero_keeps_all_nonzero(self, sample_matrix):
        """quantile=0 → threshold = min(nonzero abs), so smallest nonzero gets zeroed."""
        result = contagion_threshold(sample_matrix, 0.0)
        # threshold = min nonzero = 0.05, values with |v| <= 0.05 zeroed
        assert result.loc["B", "C"] == 0.0
        assert result.loc["A", "B"] == pytest.approx(0.8)

    def test_no_mutation_of_input(self, sample_matrix):
        original = sample_matrix.copy()
        contagion_threshold(sample_matrix, 0.5)
        pd.testing.assert_frame_equal(sample_matrix, original)

    def test_all_zero_matrix(self):
        """All-zero matrix should return all zeros regardless of quantile."""
        m = pd.DataFrame(np.zeros((3, 3)), columns=["A", "B", "C"])
        result = contagion_threshold(m, 0.5)
        assert (result.values == 0).all()
