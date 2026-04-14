import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from tqdm import tqdm

def load_data(assets, log_returns=True):
    """Load and merge data from multiple CSV files on the date column."""
    dfs = []
    for asset in assets:
        file_path = f"data/{asset}_filled.csv"
        df = pd.read_csv(file_path, parse_dates=["date"])
        dfs.append(df)

    data = dfs[0]
    for df in dfs[1:]:
        data = data.merge(df, on="date", how="inner")

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

    contagion_df = pd.DataFrame()

    for asset in tqdm(assets, desc="VAR estimation"):
        y = data[asset].iloc[n_lags:].reset_index(drop=True)

        X = lagged_df.copy()
        if not include_self:
            X = X.drop(columns=[c for c in X.columns if asset in c])

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        params = model.params.copy()
        params[model.pvalues > pvalue_threshold] = 0.0
        contagion_df[asset] = params

    return contagion_df


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