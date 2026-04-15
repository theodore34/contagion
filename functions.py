import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from tqdm import tqdm

def load_data(assets, log_returns=True, sort_by_sector=False):
    """Load and merge data from multiple CSV files on the date column.

    Parameters
    ----------
    assets : list of str
        CSV file prefixes to load (e.g. ['stock', 'crypto', 'etfs']).
    log_returns : bool
        If True, compute log-returns on numeric columns.
    sort_by_sector : bool
        If True, reorder columns by sector (from data/stock_category.xlsx)
        so that assets of the same sector are grouped together.

    Returns
    -------
    pd.DataFrame
        Data indexed by date.
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


def correlation(data, lag=0):
    """Calculate the correlation matrix for the given data with a specified lag."""
    if lag > 0:
        corr_matrix = np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
    else:
        corr_matrix = np.corrcoef(data.T)

    return corr_matrix


def corr_threshold(corr, quantile, diag=True):
    """Filter correlation matrix by quantile threshold.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix
    quantile : float
        Quantile threshold between 0 and 1 for filtering
    diag : bool, optional
        If True, consider only lower triangular part for threshold (default True)

    Returns
    -------
    np.ndarray
        Symmetric correlation matrix with 0s where values are below threshold

    Examples
    --------
    >>> corr_threshold(corr_matrix, 0.95)  # doctest: +SKIP
    """
    result = corr.copy()

    if diag:
        mask = np.tril(np.ones(corr.shape)).astype(bool)
        values_to_check = corr[mask]
    else:
        values_to_check = corr.flatten()

    thres = np.quantile(np.abs(values_to_check), quantile)
    result[np.abs(result) <= thres] = 0

    # Garder la symétrie : si un côté est 0, l'autre aussi

    return result


def var_contagion(data, n_lags=1, pvalue_threshold=0.1, include_self=True):
    """Estimate single-equation VAR and return the contagion matrix.

    For each asset j, estimates:
        r_{j,t} = alpha_j + sum_i beta_{ji} * r_{i,t-1} + eps_{j,t}

    Coefficients with p-value > pvalue_threshold are set to 0.

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns, shape (T, N). Columns = asset names.
    n_lags : int
        Number of lags (default 1).
    pvalue_threshold : float
        Significance level; coefficients above this are zeroed out.
    include_self : bool
        If True, include own lags as regressors.

    Returns
    -------
    contagion_df : pd.DataFrame
        Matrix (N*n_lags + 1, N). Rows = regressors (const + lagged assets),
        columns = target assets. Non-significant coefficients are 0.
    """
    assets = data.columns.tolist()

    # Build lag matrix once for all equations
    lagged_values = lagmat(data.values, maxlag=n_lags, trim="both")
    lag_columns = [f"{col}.L{lag}" for lag in range(1, n_lags + 1) for col in assets]
    lagged_df = pd.DataFrame(lagged_values, columns=lag_columns)

    results = {}

    for asset in tqdm(assets, desc="VAR estimation"):
        y = data[asset].iloc[n_lags:].reset_index(drop=True)

        X = lagged_df.copy()
        if not include_self:
            X = X.drop(columns=[c for c in X.columns if asset in c])

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        params = model.params.copy()
        params[model.pvalues > pvalue_threshold] = 0.0
        results[asset] = params

    return pd.DataFrame(results)


def contagion_matrix(data, n_lags=1, pvalue_threshold=0.1, lag_name="L1"):
    """Extract a square (N x N) contagion matrix for a specific lag.

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns.
    n_lags : int
        Number of lags for the VAR.
    pvalue_threshold : float
        Significance threshold.
    lag_name : str
        Which lag to extract, e.g. "L1", "L2".

    Returns
    -------
    matrix : pd.DataFrame
        Square matrix (N x N). Entry (i, j) = effect of asset i (lagged)
        on asset j. Diagonal (self-contagion) is set to 0.
    """
    raw = var_contagion(data, n_lags=n_lags, pvalue_threshold=pvalue_threshold)

    # Filter rows for the requested lag
    mask = raw.index.str.endswith(f".{lag_name}")
    matrix = raw.loc[mask].copy()
    matrix.index = [idx.rsplit(".", 1)[0] for idx in matrix.index]

    # Zero out diagonal (no self-contagion)
    for col in matrix.columns:
        if col in matrix.index:
            matrix.loc[col, col] = 0.0

    return matrix


def contagion_density(matrix):
    """Compute contagion density (%) from a square contagion matrix.

    density = #non-zero edges / #possible directed edges * 100

    Parameters
    ----------
    matrix : pd.DataFrame
        Square contagion matrix (N x N). Diagonal is ignored.

    Returns
    -------
    float
        Density as a percentage.
    """
    n = len(matrix)
    count = (matrix.values != 0).sum() - np.trace(matrix.values != 0)
    possible = n * (n - 1)
    return count / possible * 100 if possible > 0 else 0.0


def contagion_threshold(matrix, quantile):
    """Filter contagion matrix by absolute-value quantile.

    Parameters
    ----------
    matrix : pd.DataFrame
        Contagion matrix.
    quantile : float
        Quantile threshold between 0 and 1.

    Returns
    -------
    pd.DataFrame
        Filtered matrix with values below the quantile threshold set to 0.
    """
    result = matrix.copy()
    values = np.abs(matrix.values).flatten()
    thres = np.quantile(values[values > 0], quantile) if (values > 0).any() else 0
    result[np.abs(result) <= thres] = 0
    return result


def load_categories(path="data/stock_category.xlsx"):
    """Load asset categories from the Excel file.

    Maps the 12 sectors to three groups: 'stock', 'crypto', 'etf'.

    Parameters
    ----------
    path : str
        Path to the Excel file with columns 'Stocks' and 'Sectors'.

    Returns
    -------
    dict
        Mapping {asset_name: category}.
    """
    df = pd.read_excel(path)
    sector_to_cat = {"Crypto": "crypto", "US ETF": "etf"}
    return {
        row["Stocks"]: sector_to_cat.get(row["Sectors"], "stock")
        for _, row in df.iterrows()
    }


def var_contagion_masked(data, lag=1, corr_quantile=None, pvalue_threshold=0.1, mask=None):
    """Estimate VAR at a single lag, optionally masked by lag-0 correlation.

    When ``corr_quantile`` is provided, uses the lag-0 correlation matrix,
    thresholded at that quantile, as a sparsity mask: for target asset j,
    only assets i where ``mask[i, j] != 0`` are included as regressors.

    When ``corr_quantile`` is None and ``mask`` is None (default), all
    assets are used as regressors (equivalent to a standard single-lag VAR).

    For each asset j, estimates:
        r_{j,t} = alpha_j + sum_{i in M_j} beta_{ji} * r_{i, t-lag} + eps_{j,t}

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns, shape (T, N). Columns = asset names.
    lag : int
        Single lag to use (e.g., lag=3 uses only r_{i, t-3}, not lags 1-3).
    corr_quantile : float or None
        If float, quantile for thresholding the lag-0 correlation matrix
        (passed to ``corr_threshold``). If None, no masking is applied.
    pvalue_threshold : float
        Significance level; fitted coefficients with p-value above this
        are zeroed out.
    mask : np.ndarray or None
        Pre-computed sparsity mask (N x N). If provided, ``corr_quantile``
        is ignored.

    Returns
    -------
    matrix : pd.DataFrame
        Square matrix (N x N). Entry (i, j) = effect of asset i (at the
        chosen lag) on asset j. Diagonal is set to 0.
    """
    assets = data.columns.tolist()
    n_assets = len(assets)

    # Build correlation mask (or include all regressors)
    if mask is not None:
        pass
    elif corr_quantile is not None:
        corr = correlation(data.values, lag=0)
        mask = corr_threshold(corr, corr_quantile)
    else:
        mask = np.ones((n_assets, n_assets))

    # y: returns at time t, X: returns at time t-lag
    y_all = data.iloc[lag:].reset_index(drop=True)
    X_all = data.iloc[:-lag].reset_index(drop=True).values

    result = pd.DataFrame(0.0, index=assets, columns=assets)

    alpha = 1e-10

    for j_idx, asset_j in enumerate(assets):
        regressor_indices = np.where(mask[:, j_idx] != 0)[0]
        if len(regressor_indices) == 0:
            continue

        y = y_all[asset_j].values
        Xc = X_all[:, regressor_indices]
        coefs = np.linalg.solve(
            Xc.T @ Xc + alpha * np.eye(len(regressor_indices)), Xc.T @ y
        )

        for k, i_idx in enumerate(regressor_indices):
            result.iloc[i_idx, j_idx] = coefs[k]

    # No self-contagion
    for asset in assets:
        result.loc[asset, asset] = 0.0

    return result


def contagion_r2(data, matrix, lag=1, categories=None):
    """Compute R² from a contagion matrix.

    For each target asset j, uses the estimated coefficients in matrix
    to predict returns at lag, then computes R² of the predictions.

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns, shape (T, N). Columns = asset names.
    matrix : pd.DataFrame
        Contagion matrix (N x N) from contagion_matrix() or
        var_contagion_masked().
    lag : int
        Lag used to estimate the matrix.
    categories : dict or None
        Mapping {asset_name: category_name}. If provided, R² is also
        reported per category.

    Returns
    -------
    results : dict
        'total': float, mean R² across all assets.
        'per_asset': pd.Series, R² for each asset.
        'per_category': pd.Series, mean R² per category (only if
        categories is provided).
    """
    assets = data.columns.tolist()

    # y: returns at time t, X: returns at time t-lag
    y_all = data.iloc[lag:].reset_index(drop=True)
    X_all = data.iloc[:-lag].reset_index(drop=True).values

    r2_values = {}

    for j_idx, asset_j in enumerate(assets):
        y = y_all[asset_j].values

        # Prediction: sum of (coef * lagged_return) for all i
        y_pred = np.zeros_like(y)
        for i_idx, asset_i in enumerate(assets):
            coef = matrix.iloc[i_idx, j_idx]
            if coef != 0:
                y_pred += coef * X_all[:, i_idx]

        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_values[asset_j] = r2

    r2_series = pd.Series(r2_values)

    results = {
        "total": r2_series.mean(),
        "per_asset": r2_series,
    }

    if categories is not None:
        cat_series = pd.Series(categories)
        cat_series = cat_series.reindex(assets)
        results["per_category"] = r2_series.groupby(cat_series).mean()

    return results


def rolling_contagion(data, corr_quantile, asset_type, interval_size=None,
                      obs_per_regressor=2, lag=1, pvalue_threshold=0.1,
                      cache_dir="results"):
    """Compute contagion matrices and R² over fixed-size rolling windows.

    Computes the lag-0 correlation mask once on the full dataset, then
    splits the time series into non-overlapping windows of
    ``interval_size`` observations.  For each window, estimates the
    contagion matrix (via ``var_contagion_masked``) and the associated R².

    Results are cached to a pickle file.  If the file already exists it
    is loaded directly.

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns, shape (T, N). Columns = asset names.
    corr_quantile : float
        Quantile for thresholding the lag-0 correlation matrix.
    asset_type : str
        Label used in the cache filename (e.g. 'stock', 'crypto',
        'stock_crypto').
    interval_size : int or None
        Number of observations per window. If None, computed automatically
        as ``obs_per_regressor * k_max`` where k_max is the maximum
        number of regressors across all assets (from the correlation mask).
    obs_per_regressor : int
        Multiplier for automatic interval sizing. Only used when
        ``interval_size`` is None. Lower values give higher R² but
        more overfitting risk.
    lag : int
        VAR lag.
    pvalue_threshold : float
        P-value threshold for VAR coefficients.
    cache_dir : str
        Directory for pickle cache files.

    Returns
    -------
    dict
        'matrices': list of contagion DataFrames,
        'r2': list of dicts ('total', 'per_asset'),
        'intervals': list of (start_date, end_date) tuples,
        'corr_quantile': float,
        'interval_size': int,
        'k_max': int (max regressors per asset from the mask).
    """
    import os
    import pickle

    os.makedirs(cache_dir, exist_ok=True)

    # Correlation mask computed once on the full dataset
    corr = correlation(data.values, lag=0)
    mask = corr_threshold(corr, corr_quantile)

    # k_max: max number of regressors for any asset (excluding self)
    mask_no_diag = mask.copy()
    np.fill_diagonal(mask_no_diag, 0)
    k_max = int((mask_no_diag != 0).sum(axis=0).max())

    # Auto interval size
    if interval_size is None:
        interval_size = max(obs_per_regressor * k_max, 2 * lag + 1)

    filename = f"{asset_type}_q{corr_quantile}_lag{lag}_n{interval_size}.pkl"
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    n_intervals = len(data) // interval_size
    raw = data.values
    assets = data.columns.tolist()
    n_assets = len(assets)

    matrices = []
    intervals = []

    for i in tqdm(range(n_intervals), desc="Rolling contagion"):
        start = i * interval_size
        end = start + interval_size
        sub_data = data.iloc[start:end]

        matrix = var_contagion_masked(
            sub_data, lag=lag, pvalue_threshold=pvalue_threshold, mask=mask,
        )
        matrices.append(matrix)
        intervals.append((data.index[start], data.index[end - 1]))

    # R² global (comme ai-bubble) : pour chaque intervalle, predire avec
    # sa matrice, puis 1 - var(residus) / var(y_true) sur toute la serie
    y_hat_all = []
    y_true_all = []

    for i, matrix in enumerate(matrices):
        start = i * interval_size
        end = start + interval_size
        X_lag = raw[start : end - lag]
        y_true = raw[start + lag : end]
        y_hat = X_lag @ matrix.values  # (T_interval-lag, N)
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