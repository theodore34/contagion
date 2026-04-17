import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from tqdm import tqdm


def load_data(assets, log_returns=True, sort_by_sector=True):
    """Load CSV files, merge on date, optionally compute log-returns.

    Parameters
    ----------
    assets : list of str
        Asset names (files expected at data/{name}_filled.csv).
    log_returns : bool
        If True, convert prices to log-returns.
    sort_by_sector : bool
        If True, reorder columns by sector from stock_category.xlsx.
    """
    dfs = []
    for asset in assets:
        file_path = f"data/{asset}_filled.csv"
        df = pd.read_csv(file_path, parse_dates=["date"])
        dfs.append(df)

    data = dfs[0]
    for df in dfs[1:]:
        data = data.merge(df, on="date", how="inner")

    if sort_by_sector:
        cat = pd.read_excel("data/stock_category.xlsx")
        sector_order = cat.set_index("Stocks")["Sectors"]
        numeric_cols = [c for c in data.columns if c != "date"]
        sorted_cols = sorted(
            numeric_cols,
            key=lambda c: (sector_order.get(c, ""), c),
        )
        data = data[["date"] + sorted_cols]

    if log_returns:
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = np.log(data[numeric_cols] / data[numeric_cols].shift(1))
        data.dropna(inplace=True)

    data = data.set_index("date")
    return data


def load_categories(path="data/stock_category.xlsx"):
    """Load {asset: category} mapping from Excel (stock / crypto / etf).

    Parameters
    ----------
    path : str
        Path to the Excel file with columns 'Stocks' and 'Sectors'.
    """
    df = pd.read_excel(path)
    sector_to_cat = {"Crypto": "crypto", "US ETF": "etf"}
    return {
        row["Stocks"]: sector_to_cat.get(row["Sectors"], "stock")
        for _, row in df.iterrows()
    }



def correlation(data, lag=0):
    """Correlation matrix, optionally at a given lag.

    Parameters
    ----------
    data : array-like, shape (T, N)
        Time series matrix.
    lag : int
        Lag between rows. 0 = contemporaneous correlation.
    """
    if lag > 0:
        return np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
    return np.corrcoef(data.T)


def corr_threshold(corr, quantile):
    """Zero out entries below the quantile of |corr|.

    Parameters
    ----------
    corr : ndarray, shape (N, N)
        Correlation matrix.
    quantile : float
        Quantile threshold in [0, 1].
    """
    result = corr.copy()
    thres = np.quantile(np.abs(corr.flatten()), quantile)
    result[np.abs(result) <= thres] = 0
    return result



def var_contagion(data, n_lags=1):
    """OLS-based single-equation VAR.

    Parameters
    ----------
    data : DataFrame, shape (T, N)
        Log-returns with assets as columns.
    n_lags : int
        Number of lags in the VAR.

    Returns
    -------
    DataFrame, shape (N*n_lags + 1, N)
        Rows = const + lagged coefficients, columns = target assets.
    """
    assets = data.columns.tolist()

    lagged_values = lagmat(data.values, maxlag=n_lags, trim="both")
    lag_columns = [f"{col}.L{lag}" for lag in range(1, n_lags + 1) for col in assets]
    lagged_df = pd.DataFrame(lagged_values, columns=lag_columns)

    results = {}
    for asset in tqdm(assets, desc="VAR estimation"):
        y = data[asset].iloc[n_lags:].reset_index(drop=True) 
        X = sm.add_constant(lagged_df) 
        model = sm.OLS(y, X).fit()  # Fit OLS regression: Y = X * beta + error
        results[asset] = model.params.copy()

    return pd.DataFrame(results)


def var_contagion_masked(data, lag=1, corr_quantile=None, mask=None):
    """OLS VAR at a single lag with optional sparsity mask and constant.

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    lag : int
        Lag order.
    corr_quantile : float or None
        If set, build mask from correlation quantile.
    mask : ndarray or None
        Pre-computed (N, N) sparsity mask. Overrides corr_quantile.

    Returns
    -------
    DataFrame (N+1 x N)
        Rows = const + assets, columns = target assets.
    """
    assets = data.columns.tolist()
    n_assets = len(assets)

    if mask is not None:
        pass
    elif corr_quantile is not None:
        corr = correlation(data.values, lag=0)
        mask = corr_threshold(corr, corr_quantile)
    else:
        mask = np.ones((n_assets, n_assets))

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
        Xc = np.column_stack([np.ones(len(Xc)), Xc])  # ajout constante
        coefs = np.linalg.lstsq(Xc, y, rcond=None)[0]

        result.loc["const", asset_j] = coefs[0]
        for k, i_idx in enumerate(regressor_indices):
            result.iloc[i_idx + 1, j_idx] = coefs[k + 1]

    return result


def contagion_r2(data, matrix, lag=1, categories=None):
    """R² from a contagion matrix.

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    matrix : DataFrame (N+1 x N)
        Contagion coefficient matrix (with const row).
    lag : int
        Lag used to build the matrix.
    categories : dict or None
        {asset: category} mapping. If set, adds 'per_category' R².

    Returns
    -------
    dict
        Keys: 'total', 'per_asset', optionally 'per_category'.
    """
    assets = data.columns.tolist()
    has_const = "const" in matrix.index

    y_all = data.iloc[lag:].reset_index(drop=True)
    X_all = data.iloc[:-lag].reset_index(drop=True).values

    r2_values = {}
    for j_idx, asset_j in enumerate(assets):
        y = y_all[asset_j].values
        y_pred = np.zeros_like(y)

        if has_const:
            y_pred += matrix.loc["const", asset_j]

        for i_idx, asset_i in enumerate(assets):
            coef = matrix.loc[asset_i, asset_j]
            if coef != 0:
                y_pred += coef * X_all[:, i_idx]

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_values[asset_j] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    r2_series = pd.Series(r2_values)
    results = {"total": r2_series.mean(), "per_asset": r2_series}

    if categories is not None:
        cat_series = pd.Series(categories).reindex(assets)
        results["per_category"] = r2_series.groupby(cat_series).mean()

    return results


def rolling_contagion(data, corr_quantile, asset_type, interval_size=None,
                      obs_per_regressor=2, lag=1, cache_dir="results"):
    """Contagion matrices over non-overlapping rolling windows (cached).

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    corr_quantile : float
        Quantile for the correlation sparsity mask.
    asset_type : str
        Label for cache filename (e.g. 'stock', 'crypto').
    interval_size : int or None
        Number of (X, y) observations used to fit each window. The raw window
        spans `interval_size + lag` rows; the first `lag` rows are consumed
        to build the lagged design matrix. Auto-computed from k_max if None.
    obs_per_regressor : int
        Min observations per regressor for auto interval_size.
    lag : int
        Lag order.
    cache_dir : str
        Directory for pickle cache.

    Returns
    -------
    dict
        Keys: 'matrices', 'r2_per_asset', 'r2_total', 'intervals',
        'corr_quantile', 'interval_size', 'k_max'.
    """
    os.makedirs(cache_dir, exist_ok=True)

    #calcul masque
    corr = correlation(data.values, lag=0)
    mask = corr_threshold(corr, corr_quantile)

    k_max = int((mask != 0).sum(axis=0).max())

    # interval_size = nombre d'observations utilisables pour le fit.
    # La fenêtre brute vaut interval_size + lag.
    if interval_size is None:
        interval_size = max(obs_per_regressor * k_max, 1)

    window = interval_size + lag

    filename = f"{asset_type}_q{corr_quantile}_lag{lag}_n{interval_size}.pkl"
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    n_intervals = len(data) // window
    raw = data.values
    assets = data.columns.tolist()

    matrices = []
    intervals = []

    for i in tqdm(range(n_intervals), desc="Rolling contagion"):
        start = i * window
        end = start + window
        sub_data = data.iloc[start:end]
        matrix = var_contagion_masked(sub_data, lag=lag, mask=mask)
        matrices.append(matrix)
        intervals.append((data.index[start], data.index[end - 1]))

    # R² global -> concaténation des prédictions
    y_hat_all = []
    y_true_all = []
    for i, matrix in enumerate(matrices):
        start = i * window
        end = start + window
        X_lag = raw[start : end - lag]
        y_true = raw[start + lag : end]
        coef = matrix.loc[assets].values
        const = matrix.loc["const"].values
        y_hat = X_lag @ coef + const
        y_hat_all.append(y_hat)
        y_true_all.append(y_true)

    y_hat_all = np.concatenate(y_hat_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    residus = y_true_all - y_hat_all
    r2_per_asset = pd.Series(
        1 - residus.var(axis=0) / y_true_all.var(axis=0),
        index=assets,
    )

    results = {
        "matrices": matrices,
        "r2_per_asset": r2_per_asset,
        "r2_total": float(r2_per_asset.mean()),
        "intervals": intervals,
        "corr_quantile": corr_quantile,
        "interval_size": interval_size,
        "k_max": k_max,
    }

    with open(filepath, "wb") as f:
        pickle.dump(results, f)

    return results


def activation_frequency(data, corr_quantile, asset_type, interval_size=None,
                         lag=1, binarization_quantile=0.8, eps=1e-6,
                         plot=True, **kwargs):
    """Binarize rolling contagion matrices and compute per-link activation frequency.

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    corr_quantile : float
        Quantile for sparsity mask.
    asset_type : str
        Label for cache.
    interval_size : int or None
        Window size (auto if None).
    lag : int
        Lag order.
    binarization_quantile : float
        Per-matrix quantile threshold for binarization.
    eps : float
        Values below eps are treated as zero.
    plot : bool
        If True, display heatmap.

    Returns
    -------
    dict
        Keys: 'freq' (N,N), 'binary' (K,N,N), 'rolling'.
    """
    res = rolling_contagion(data, corr_quantile=corr_quantile,
                            asset_type=asset_type,
                            interval_size=interval_size, lag=lag, **kwargs)
    assets = data.columns.tolist()
    mat_arr = np.array([m.loc[assets].values for m in res["matrices"]])

    for m in mat_arr:
        np.fill_diagonal(m, 0)

    mat_binary = []
    for m in mat_arr:
        nonzero = m[np.abs(m) > eps]
        if len(nonzero) > 0:
            threshold = np.quantile(np.abs(nonzero), binarization_quantile)
            mat_binary.append(np.where(np.abs(m) >= threshold, 1, 0))
        else:
            mat_binary.append(np.zeros_like(m, dtype=int))
    mat_binary = np.array(mat_binary)

    freq = np.mean(mat_binary, axis=0)

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(freq, vmin=0, vmax=1, ax=ax)
        ax.set_title(
            f"Freq. activation "
            f"(corr_q={corr_quantile}, bin_q={binarization_quantile}, "
            f"interval={res['interval_size']})"
        )
        plt.tight_layout()
        plt.show()

    return {"freq": freq, "binary": mat_binary, "rolling": res}


def mean_magnitude(data, corr_quantile, asset_type, interval_size=None,
                   lag=1, eps=1e-6, plot=True, **kwargs):
    """Mean coefficient value across rolling windows (signed).

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    corr_quantile : float
        Quantile for sparsity mask.
    asset_type : str
        Label for cache.
    interval_size : int or None
        Window size (auto if None).
    lag : int
        Lag order.
    eps : float
        Values below eps are treated as zero.
    plot : bool
        If True, display heatmap and stats.

    Returns
    -------
    dict
        Keys: 'magnitude' (N,N), 'rolling'.
    """
    res = rolling_contagion(data, corr_quantile=corr_quantile,
                            asset_type=asset_type,
                            interval_size=interval_size, lag=lag, **kwargs)
    assets = data.columns.tolist()
    mat_arr = np.array([m.loc[assets].values for m in res["matrices"]])

    for m in mat_arr:
        np.fill_diagonal(m, 0)

    magnitude = np.mean(mat_arr, axis=0)

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(magnitude, center=0, ax=ax)
        ax.set_title(
            f"Magnitude moyenne coeff "
            f"(lag={lag}, corr_q={corr_quantile}, "
            f"interval={res['interval_size']})"
        )
        plt.tight_layout()
        plt.show()
        print(f"Magnitude moyenne globale : {magnitude.mean():.6f}")
        print(f"Min / Max : {magnitude.min():.6f} / {magnitude.max():.6f}")
        print(f"Liens nuls (|m| < eps) : {(np.abs(magnitude) < eps).sum()}")

    return {"magnitude": magnitude, "rolling": res}
