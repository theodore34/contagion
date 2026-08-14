"""Build contagion matrices from returns / correlations.

Four families of networks then feed the SIS dynamics:
  - PMFG       : Planar Maximally Filtered Graph weighted by |C_ij|
  - VAR(1)     : coefficients of a 1-lag VAR
  - VAR(13)    : coefficients of a 13-lag VAR
  - Corr thr   : correlation thresholded by a quantile of |C_ij|

Every matrix meant for the SIS goes through :func:`prepare_for_sis`
(absolute value, transpose, zero diagonal).
"""
import networkx as nx
import numpy as np


def correlation(data, lag=0):
    """Correlation matrix, optionally at a given lag.

    Parameters
    ----------
    data : array-like, shape (T, N)
        Matrix of time series.
    lag : int
        Lag between rows. 0 = contemporaneous correlation.

    Returns
    -------
    ndarray, shape (N, N)
        Correlation between ``data[:-lag]`` and ``data[lag:]`` (full if lag=0).
    """
    if lag > 0:
        return np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
    return np.corrcoef(data.T)


def build_pmfg(C_abs):
    """Planar Maximally Filtered Graph weighted by |C| (3(n-2) edges max).

    Edges are added by decreasing weight as long as the graph stays planar,
    up to 3(n-2) edges.

    Parameters
    ----------
    C_abs : ndarray, shape (n, n)
        Positive similarity matrix (typically |C_ij|).

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric weighted adjacency matrix of the PMFG.
    """
    n = C_abs.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = sorted(((a, b, C_abs[a, b]) for a in range(n) for b in range(a + 1, n)),
                   key=lambda e: -e[2])
    A, added = np.zeros((n, n)), 0
    for a, b, w in edges:
        if added >= 3 * (n - 2):
            break
        G.add_edge(a, b)
        if nx.check_planarity(G)[0]:
            A[a, b] = A[b, a] = w
            added += 1
        else:
            G.remove_edge(a, b)
    return A


def fit_var(arr, lag):
    """VAR(lag) coefficients: regression of ``arr[lag:]`` on ``arr[:-lag]``.

    Parameters
    ----------
    arr : ndarray, shape (T, N)
        Time series (log-returns).
    lag : int
        Number of lags.

    Returns
    -------
    ndarray, shape (N, N)
        Coefficient block (the constant term is dropped).
    """
    X = np.column_stack([np.ones(len(arr) - lag), arr[:-lag]])
    B, *_ = np.linalg.lstsq(X, arr[lag:], rcond=None)
    return B[1:]


def prepare_for_sis(A_in):
    """Contagion matrix ready for the SIS: |A| transposed, zero diagonal.

    Parameters
    ----------
    A_in : ndarray, shape (N, N)
        Raw contagion matrix (may be signed and asymmetric).

    Returns
    -------
    ndarray, shape (N, N)
        ``|A_in|`` transposed, with zero diagonal (a copy).
    """
    A = np.abs(A_in.T)
    np.fill_diagonal(A, 0)
    return A


def build_filtered_corr(C, q, mask_off):
    """Thresholded correlation ready for the SIS: |C_ij| above quantile q.

    Parameters
    ----------
    C : ndarray, shape (N, N)
        Absolute-value correlation matrix (diagonal unused).
    q : float
        Quantile (over off-diagonal terms) below which entries are zeroed.
    mask_off : ndarray of bool, shape (N, N)
        Off-diagonal mask (``~np.eye(N, bool)``).

    Returns
    -------
    ndarray, shape (N, N)
        Thresholded matrix, passed through :func:`prepare_for_sis`.
    """
    A = np.where(C >= np.quantile(C[mask_off], q), C, 0.0)
    np.fill_diagonal(A, 0)
    return prepare_for_sis(A)
