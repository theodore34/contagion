import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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