"""Unit tests for the code/ package — run with: pytest tests/test_code.py -v"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))

from data import load_data
from networks import correlation, fit_var, prepare_for_sis, build_filtered_corr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_var1(coefs, n_obs=3000, noise_std=0.01, seed=0, assets=("A", "B", "C")):
    """Generate a VAR(1) series with known coefficients: target_t = coef * src_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    assets = sorted(assets)
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    B = np.zeros((n, n))
    for (src, tgt), val in coefs.items():
        B[idx[src], idx[tgt]] = val
    data = np.zeros((n_obs, n))
    for t in range(1, n_obs):
        data[t] = B.T @ data[t - 1] + rng.normal(0, noise_std, n)
    return pd.DataFrame(data, columns=assets)


# ---------------------------------------------------------------------------
# data.load_data
# ---------------------------------------------------------------------------

def _mock_csv(dfs):
    def _read(path, **kwargs):
        asset = os.path.basename(path).replace("_filled.csv", "")
        return dfs[asset].copy()
    return _read


class TestLoadData:

    def test_log_returns_of_doubling_series(self):
        """log(2^t / 2^{t-1}) = log(2) for every step."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"date": dates, "price": [1.0, 2.0, 4.0, 8.0, 16.0]})
        with patch("pandas.read_csv", side_effect=_mock_csv({"A": df})):
            result = load_data(["A"])
        np.testing.assert_allclose(result["price"].values, np.log(2.0))
        assert len(result) == 4  # one row dropped after shift

    def test_inner_join_keeps_only_overlap(self):
        """Only the 3 common dates should remain."""
        dates_a = pd.date_range("2023-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2023-01-03", periods=5, freq="D")
        df_a = pd.DataFrame({"date": dates_a, "pa": [1.0] * 5})
        df_b = pd.DataFrame({"date": dates_b, "pb": [1.0] * 5})
        with patch("pandas.read_csv", side_effect=_mock_csv({"A": df_a, "B": df_b})):
            result = load_data(["A", "B"], log_returns=False)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# networks.correlation
# ---------------------------------------------------------------------------

class TestCorrelation:

    def test_diagonal_ones_and_symmetry(self):
        data = np.random.default_rng(0).standard_normal((100, 3))
        C = correlation(data, lag=0)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_identical_series_correlation_one(self):
        x = np.random.default_rng(0).standard_normal(80)
        C = correlation(np.column_stack([x, x]), lag=0)
        np.testing.assert_allclose(C, np.ones((2, 2)), atol=1e-10)

    def test_lag_aligns_shifted_series(self):
        """A series shifted by k correlates perfectly with itself at lag=k."""
        base = np.random.default_rng(42).standard_normal(100)
        for k in (1, 3, 5):
            data = np.zeros((100, 2))
            data[:, 0] = base
            data[k:, 1] = base[:-k]
            C = correlation(data, lag=k)
            assert C[0, 1] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# networks.fit_var
# ---------------------------------------------------------------------------

class TestFitVar:

    def test_output_shape(self):
        df = _make_var1({("A", "B"): 0.3}, n_obs=200)
        B = fit_var(df.values, 1)
        n = len(df.columns)
        assert B.shape == (n, n)

    def test_recovers_known_coefficient(self):
        """A -> B at 0.5 lands in B[0, 1] (assets sorted: A=0, B=1)."""
        df = _make_var1({("A", "B"): 0.5})
        B = fit_var(df.values, 1)
        assert B[0, 1] == pytest.approx(0.5, abs=0.05)
        assert B[0, 2] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# networks.prepare_for_sis
# ---------------------------------------------------------------------------

class TestPrepareForSis:

    def test_abs_transpose_zero_diagonal(self):
        A_in = np.array([[0.9, -0.5], [0.2, 0.7]])
        A = prepare_for_sis(A_in)
        np.testing.assert_allclose(A, [[0.0, 0.2], [0.5, 0.0]])

    def test_does_not_mutate_input(self):
        A_in = np.array([[0.9, -0.5], [0.2, 0.7]])
        original = A_in.copy()
        prepare_for_sis(A_in)
        np.testing.assert_array_equal(A_in, original)


# ---------------------------------------------------------------------------
# networks.build_filtered_corr
# ---------------------------------------------------------------------------

class TestBuildFilteredCorr:

    def test_known_example(self):
        C = np.array([
            [0.0, 0.8, 0.2],
            [0.8, 0.0, 0.5],
            [0.2, 0.5, 0.0],
        ])
        mask_off = ~np.eye(3, dtype=bool)
        thres = np.quantile(C[mask_off], 0.5)
        expected = np.where(C >= thres, C, 0.0)
        np.fill_diagonal(expected, 0)
        result = build_filtered_corr(C, 0.5, mask_off)
        np.testing.assert_allclose(result, expected.T, atol=1e-12)

    def test_preserves_symmetry(self):
        C = np.array([
            [0.0, 0.9, 0.3, 0.1],
            [0.9, 0.0, 0.5, 0.2],
            [0.3, 0.5, 0.0, 0.7],
            [0.1, 0.2, 0.7, 0.0],
        ])
        mask_off = ~np.eye(4, dtype=bool)
        result = build_filtered_corr(C, 0.5, mask_off)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_does_not_mutate_input(self):
        C = np.array([[0.0, 0.5], [0.5, 0.0]])
        original = C.copy()
        build_filtered_corr(C, 0.5, ~np.eye(2, dtype=bool))
        np.testing.assert_array_equal(C, original)
