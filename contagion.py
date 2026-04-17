"""
Utilisation: 

res = rolling_contagion(["stock_filled.csv", "crypto_filled.csv"], corr_quantile = 0.8, asset_type="stocks-crypto", interval_size=100, lag=2)
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_log_returns(csv_paths, date_col="date"):
    """Read one or more price CSVs, inner-join on date, return log-returns."""
    if isinstance(csv_paths, (str, os.PathLike)):
        csv_paths = [csv_paths]

    merged = None
    for path in csv_paths:
        df = pd.read_csv(path, parse_dates=[date_col])
        merged = df if merged is None else merged.merge(df, on=date_col, how="inner")

    merged = merged.set_index(date_col)
    numeric = merged.select_dtypes(include=np.number)
    returns = np.log(numeric / numeric.shift(1))
    return returns.dropna()


def _corr_threshold(corr, quantile):
    """Zero out entries at or below the |corr| quantile."""
    result = corr.copy()
    thres = np.quantile(np.abs(corr.flatten()), quantile)
    result[np.abs(result) <= thres] = 0
    return result


def _var_contagion_masked(data, lag, mask):
    """OLS VAR with sparsity mask and constant; returns (N+1, N) DataFrame."""
    assets = data.columns.tolist()
    y_all = data.iloc[lag:].reset_index(drop=True)
    X_all = data.iloc[:-lag].reset_index(drop=True).values

    result = pd.DataFrame(0.0, index=["const"] + assets, columns=assets)

    for j_idx, asset_j in enumerate(assets):
        regressor_indices = np.where(mask[:, j_idx] != 0)[0]
        y = y_all[asset_j].values

        if len(regressor_indices) == 0:
            result.loc["const", asset_j] = y.mean()
            continue

        Xc = X_all[:, regressor_indices]
        Xc = np.column_stack([np.ones(len(Xc)), Xc])
        coefs = np.linalg.lstsq(Xc, y, rcond=None)[0]

        result.loc["const", asset_j] = coefs[0]
        for k, i_idx in enumerate(regressor_indices):
            result.iloc[i_idx + 1, j_idx] = coefs[k + 1]

    return result


def rolling_contagion(csv_path, corr_quantile, asset_type, interval_size=None,
                      obs_per_regressor=2, lag=1, cache_dir="results",
                      date_col="date"):
    """Contagion matrices over non-overlapping rolling windows from CSV(s).

    Parameters
    ----------
    csv_path : str or list of str
        One CSV path, or a list of CSV paths that share the date column
        (inner-joined on date).
    corr_quantile : float
        Quantile for the correlation sparsity mask.
    asset_type : str
        Label for cache filename (e.g. 'stock', 'crypto').
    interval_size : int or None
        Number of (X, y) observations used to fit each window. The raw window
        spans `interval_size + lag` rows. Auto-computed from k_max if None.
    obs_per_regressor : int
        Min observations per regressor for auto interval_size.
    lag : int
        Lag order.
    cache_dir : str
        Directory for pickle cache.
    date_col : str
        Name of the date column in the CSV.

    Returns
    -------
    dict
        Keys: 'matrices', 'r2_per_asset', 'r2_total', 'intervals',
        'corr_quantile', 'interval_size', 'k_max', 'corr'.
    """
    os.makedirs(cache_dir, exist_ok=True)

    data = _load_log_returns(csv_path, date_col=date_col)

    corr = np.corrcoef(data.values.T)
    mask = _corr_threshold(corr, corr_quantile)
    k_max = int((mask != 0).sum(axis=0).max())

    if interval_size is None:
        interval_size = max(obs_per_regressor * k_max, 1)

    window = interval_size + lag

    filename = f"{asset_type}_q{corr_quantile}_lag{lag}_n{interval_size}.pkl"
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            res =  pickle.load(f)
            print("R^2 moyen:")
            print(res["r2_total"])
            return res

    n_intervals = len(data) // window
    raw = data.values
    assets = data.columns.tolist()

    matrices = []
    intervals = []
    for i in tqdm(range(n_intervals), desc="Rolling contagion"):
        start = i * window
        end = start + window
        sub = data.iloc[start:end]
        matrices.append(_var_contagion_masked(sub, lag=lag, mask=mask))
        intervals.append((data.index[start], data.index[end - 1]))

    y_hat_all, y_true_all = [], []
    for i, matrix in enumerate(matrices):
        start = i * window
        end = start + window
        X_lag = raw[start:end - lag]
        y_true = raw[start + lag:end]
        coef = matrix.loc[assets].values
        const = matrix.loc["const"].values
        y_hat_all.append(X_lag @ coef + const)
        y_true_all.append(y_true)

    y_hat_all = np.concatenate(y_hat_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    residus = y_true_all - y_hat_all
    r2_per_asset = pd.Series(
        1 - residus.var(axis=0) / y_true_all.var(axis=0),
        index=assets,
    )

    corr_df = pd.DataFrame(corr, index=assets, columns=assets)

    results = {
        "matrices": matrices,
        "r2_per_asset": r2_per_asset,
        "r2_total": float(r2_per_asset.mean()),
        "intervals": intervals,
        "corr_quantile": corr_quantile,
        "interval_size": interval_size,
        "k_max": k_max,
        "corr": corr_df,
    }

    with open(filepath, "wb") as f:
        pickle.dump(results, f)

    print("R^2 moyen:")
    print(float(r2_per_asset.mean()))

    return results
