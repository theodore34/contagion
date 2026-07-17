"""Construction des matrices de contagion à partir des rendements / corrélations.

Quatre familles de réseaux alimentent ensuite la dynamique SIS :
  - PMFG       : Planar Maximally Filtered Graph pondéré par |C_ij|
  - VAR(1)     : coefficients d'un VAR à 1 retard
  - VAR(13)    : coefficients d'un VAR à 13 retards
  - Corr thr   : corrélation seuillée par un quantile de |C_ij|

Toutes les matrices destinées au SIS passent par :func:`prepare_for_sis`
(valeur absolue, transposition, diagonale nulle).
"""
import networkx as nx
import numpy as np


def correlation(data, lag=0):
    """Matrice de corrélation, éventuellement à un retard donné.

    Parameters
    ----------
    data : array-like, shape (T, N)
        Matrice de séries temporelles.
    lag : int
        Retard entre lignes. 0 = corrélation contemporaine.

    Returns
    -------
    ndarray, shape (N, N)
        Corrélation entre ``data[:-lag]`` et ``data[lag:]`` (pleine si lag=0).
    """
    if lag > 0:
        return np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
    return np.corrcoef(data.T)


def build_pmfg(C_abs):
    """Planar Maximally Filtered Graph pondéré par |C| (3(n-2) arêtes max).

    Les arêtes sont ajoutées par poids décroissant tant que le graphe reste
    planaire, jusqu'à atteindre 3(n-2) arêtes.

    Parameters
    ----------
    C_abs : ndarray, shape (n, n)
        Matrice de similarité positive (typiquement |C_ij|).

    Returns
    -------
    ndarray, shape (n, n)
        Matrice d'adjacence pondérée symétrique du PMFG.
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
    """Coefficients VAR(lag) : régression de ``arr[lag:]`` sur ``arr[:-lag]``.

    Parameters
    ----------
    arr : ndarray, shape (T, N)
        Série temporelle (log-rendements).
    lag : int
        Nombre de retards.

    Returns
    -------
    ndarray, shape (N, N)
        Bloc de coefficients (la constante est retirée).
    """
    X = np.column_stack([np.ones(len(arr) - lag), arr[:-lag]])
    B, *_ = np.linalg.lstsq(X, arr[lag:], rcond=None)
    return B[1:]


def prepare_for_sis(A_in):
    """Matrice de contagion prête pour le SIS : |A| transposée, diagonale nulle.

    Parameters
    ----------
    A_in : ndarray, shape (N, N)
        Matrice de contagion brute (peut être signée et asymétrique).

    Returns
    -------
    ndarray, shape (N, N)
        ``|A_in|`` transposée, à diagonale nulle (copie).
    """
    A = np.abs(A_in.T)
    np.fill_diagonal(A, 0)
    return A


def build_filtered_corr(C, q, mask_off):
    """Corrélation seuillée prête pour le SIS : |C_ij| au-dessus du quantile q.

    Parameters
    ----------
    C : ndarray, shape (N, N)
        Matrice de corrélation en valeur absolue (diagonale non utilisée).
    q : float
        Quantile (sur les termes hors diagonale) en deçà duquel on annule.
    mask_off : ndarray of bool, shape (N, N)
        Masque hors-diagonale (``~np.eye(N, bool)``).

    Returns
    -------
    ndarray, shape (N, N)
        Matrice seuillée, passée par :func:`prepare_for_sis`.
    """
    A = np.where(C >= np.quantile(C[mask_off], q), C, 0.0)
    np.fill_diagonal(A, 0)
    return prepare_for_sis(A)
