"""Chargement des données et calculs de base (mis en cache une fois pour toutes).

`build_context()` est le point d'entrée commun à tous les scripts : il renvoie un
objet portant les rendements, les tableaux journaliers (⟨|C_ij|⟩ et E_i), la
corrélation pleine période, les quatre matrices de contagion prêtes pour le SIS,
et les métadonnées (noms d'actifs, N, masque hors-diagonale).

Le bloc lourd `compute_base` (boucle journalière + PMFG + VAR) est mis en cache
sous la signature ``N{N}T{T}`` : il ne dépend que des données, pas des
hyperparamètres, donc il survit aux changements de réglage.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
from cache import disk_cache
from networks import build_filtered_corr, build_pmfg, fit_var, prepare_for_sis


def load_data(assets, log_returns=True, sort_by_sector=True):
    """Charge les CSV des actifs, fusionne sur la date, rendements optionnels.

    Parameters
    ----------
    assets : list of str
        Noms d'actifs (fichiers attendus : ``DATA_DIR/{nom}_filled.csv``).
    log_returns : bool
        Si True, convertit les prix en log-rendements.
    sort_by_sector : bool
        Si True, réordonne les colonnes par secteur (stock_category.xlsx).

    Returns
    -------
    DataFrame, shape (T, N)
        Index = horodatage, colonnes = actifs.
    """
    dfs = [pd.read_csv(config.DATA_DIR / f"{a}_filled.csv", parse_dates=["date"])
           for a in assets]
    data = dfs[0]
    for df in dfs[1:]:
        data = data.merge(df, on="date", how="inner")

    if sort_by_sector:
        cat = pd.read_excel(config.DATA_DIR / "stock_category.xlsx")
        sector_order = cat.set_index("Stocks")["Sectors"]
        numeric_cols = [c for c in data.columns if c != "date"]
        sorted_cols = sorted(numeric_cols, key=lambda c: (sector_order.get(c, ""), c))
        data = data[["date"] + sorted_cols]

    if log_returns:
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = np.log(data[numeric_cols] / data[numeric_cols].shift(1))
        data = data.dropna()

    return data.set_index("date")


def load_returns():
    """Log-rendements intraday des actifs du pipeline (voir :func:`load_data`).

    Returns
    -------
    DataFrame, shape (T, N)
        Index = horodatage, colonnes = actifs (ordonnées par secteur puis nom).
    """
    return load_data(config.ASSETS, log_returns=config.LOG_RETURNS,
                     sort_by_sector=config.SORT_BY_SECTOR)


def _compute_base(data, N, mask_off):
    """Tableaux journaliers + réseaux pleine période (calcul lourd, mis en cache).

    Returns
    -------
    dict
        ``all_days`` (jours), ``mc_all`` (⟨|C_ij|⟩ journalier),
        ``E_daily_all`` (N x jours, E_i journalier), ``C_full`` (|corr| pleine
        période, diag. nulle), ``A_pmfg`` / ``A_var1`` / ``A_var13`` (réseaux).
    """
    da = data.copy()
    da["day"] = da.index.date
    all_days = np.array(sorted(da["day"].unique()))
    mc_all = np.full(len(all_days), np.nan)
    E_daily_all = np.full((N, len(all_days)), np.nan)
    for k, dd in enumerate(all_days):
        chunk = da[da["day"] == dd].drop(columns=["day"]).values
        if len(chunk) < 3:
            continue
        C = np.abs(np.nan_to_num(np.corrcoef(chunk.T), nan=0.0))
        np.fill_diagonal(C, 0)
        mc_all[k] = C[mask_off].mean()
        E_daily_all[:, k] = C.sum(axis=1) / (N - 1)
    C_full = np.abs(np.nan_to_num(np.corrcoef(data.values.T), nan=0.0))
    np.fill_diagonal(C_full, 0)
    A_pmfg = build_pmfg(C_full.copy())
    A_var1 = np.abs(fit_var(data.values, 1).T)
    np.fill_diagonal(A_var1, 0)
    A_var13 = np.abs(fit_var(data.values, 13).T)
    np.fill_diagonal(A_var13, 0)
    return dict(all_days=all_days, mc_all=mc_all, E_daily_all=E_daily_all,
                C_full=C_full, A_pmfg=A_pmfg, A_var1=A_var1, A_var13=A_var13)


def build_sis_matrices(base, mask_off):
    """Les quatre matrices de contagion prêtes pour le SIS (dict par méthode).

    Parameters
    ----------
    base : dict
        Sortie de :func:`_compute_base`.
    mask_off : ndarray of bool
        Masque hors-diagonale.

    Returns
    -------
    dict
        ``{'PMFG', 'VAR(1)', 'VAR(13)', 'Corr thr'}`` -> ndarray (N, N).
    """
    return {
        "PMFG": base["A_pmfg"],
        "VAR(1)": prepare_for_sis(base["A_var1"]),
        "VAR(13)": prepare_for_sis(base["A_var13"]),
        "Corr thr": build_filtered_corr(base["C_full"], config.CORR_THRESHOLD, mask_off),
    }


def build_context():
    """Charge données + base + matrices et renvoie le contexte partagé.

    Returns
    -------
    SimpleNamespace
        Champs : ``data, asset_names, N, mask_off, all_days, mc_all,
        E_daily_all, C_full, A_pmfg, A_var1, A_var13, mean_mc, A_sis``.
    """
    data = load_returns()
    asset_names = data.columns.tolist()
    N = len(asset_names)
    mask_off = ~np.eye(N, dtype=bool)
    print(f"N = {N} actifs, T = {len(data)} obs ; {data.index[0]} -> {data.index[-1]}")

    base = disk_cache("base", f"N{N}T{len(data)}",
                      lambda: _compute_base(data, N, mask_off))
    A_sis = build_sis_matrices(base, mask_off)
    mean_mc = float(np.nanmean(base["mc_all"]))
    print(f"{len(base['all_days'])} jours ; <|C_ij|> moyen = {mean_mc:.4f}")

    return SimpleNamespace(data=data, asset_names=asset_names, N=N, mask_off=mask_off,
                           mean_mc=mean_mc, A_sis=A_sis, **base)
