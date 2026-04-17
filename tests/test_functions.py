"""Unit tests for functions.py — run with: pytest tests/test_functions.py -v"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import (
    load_data, correlation, corr_threshold,
    var_contagion, var_contagion_masked,
    contagion_r2, rolling_contagion,
)


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
# load_data
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
# correlation
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
# corr_threshold
# ---------------------------------------------------------------------------

class TestCorrThreshold:

    def test_known_example(self):
        corr = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ])
        thres = np.quantile(np.abs(corr.flatten()), 0.5)
        expected = np.where(np.abs(corr) <= thres, 0, corr)
        result = corr_threshold(corr, 0.5)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_preserves_symmetry(self):
        corr = np.array([
            [1.0, 0.9, 0.3, 0.1],
            [0.9, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ])
        result = corr_threshold(corr, 0.5)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_does_not_mutate_input(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        original = corr.copy()
        corr_threshold(corr, 0.5)
        np.testing.assert_array_equal(corr, original)


# ---------------------------------------------------------------------------
# var_contagion
# ---------------------------------------------------------------------------

class TestVarContagion:

    def test_output_shape(self):
        df = _make_var1({("A", "B"): 0.3}, n_obs=200)
        result = var_contagion(df, n_lags=1)
        n = len(df.columns)
        assert result.shape == (n + 1, n)  # const + N lagged coefs
        assert list(result.columns) == list(df.columns)

    def test_recovers_known_coefficient(self):
        df = _make_var1({("A", "B"): 0.5})
        result = var_contagion(df, n_lags=1)
        assert result.loc["A.L1", "B"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# var_contagion_masked
# ---------------------------------------------------------------------------

class TestVarContagionMasked:

    def test_output_shape_and_index(self):
        df = _make_var1({("A", "B"): 0.3}, n_obs=200)
        result = var_contagion_masked(df, lag=1)
        n = len(df.columns)
        assert result.shape == (n + 1, n)
        assert list(result.index) == ["const"] + list(df.columns)
        assert list(result.columns) == list(df.columns)

    def test_recovers_known_coefficient(self):
        df = _make_var1({("A", "B"): 0.5})
        result = var_contagion_masked(df, lag=1)
        assert result.loc["A", "B"] == pytest.approx(0.5, abs=0.05)

    def test_mask_zero_entries_are_zero(self):
        df = _make_var1({("A", "B"): 0.5})
        mask = np.ones((3, 3))
        mask[0, 1] = 0  # block A -> B
        result = var_contagion_masked(df, lag=1, mask=mask)
        assert result.loc["A", "B"] == 0.0


# ---------------------------------------------------------------------------
# contagion_r2
# ---------------------------------------------------------------------------

class TestContagionR2:

    def test_r2_positive_for_predictable_asset(self):
        """B depends on A → R²(B) > 0; C is independent → R²(C) ≈ 0."""
        df = _make_var1({("A", "B"): 0.5})
        matrix = var_contagion_masked(df, lag=1)
        r2 = contagion_r2(df, matrix, lag=1)
        assert 0.0 <= r2["total"] <= 1.0
        assert r2["per_asset"]["B"] > 0.1
        assert r2["per_asset"]["C"] < 0.05

    def test_per_category_aggregation(self):
        df = _make_var1({("A", "B"): 0.5})
        matrix = var_contagion_masked(df, lag=1)
        cats = {"A": "stock", "B": "stock", "C": "crypto"}
        r2 = contagion_r2(df, matrix, lag=1, categories=cats)
        assert set(r2["per_category"].index) == {"stock", "crypto"}


# ---------------------------------------------------------------------------
# rolling_contagion
# ---------------------------------------------------------------------------

class TestRollingContagion:

    def test_shapes_and_intervals(self, tmp_path):
        df = _make_var1({("A", "B"): 0.3}, n_obs=200)
        res = rolling_contagion(
            df, corr_quantile=0.5, asset_type="test",
            interval_size=50, cache_dir=str(tmp_path),
        )
        assert len(res["matrices"]) == 4  # 200 // 50
        for m in res["matrices"]:
            assert m.shape == (4, 3)
            assert m.index[0] == "const"

    def test_cache_roundtrip(self, tmp_path):
        """Second call loads from cache and returns identical matrices."""
        df = _make_var1({("A", "B"): 0.3}, n_obs=200)
        kwargs = dict(corr_quantile=0.5, asset_type="test",
                      interval_size=50, cache_dir=str(tmp_path))
        res1 = rolling_contagion(df, **kwargs)
        res2 = rolling_contagion(df, **kwargs)
        assert (tmp_path / "test_q0.5_lag1_n50.pkl").exists()
        for m1, m2 in zip(res1["matrices"], res2["matrices"]):
            pd.testing.assert_frame_equal(m1, m2)
