"""Lecture des fits SIS : tableaux R², sélection des périodes et des courbes.

À partir des dictionnaires de fits produits par :mod:`sis` (``{(pa, pb):
(cache, fits)}``), ce module :
  - construit les tableaux récapitulatifs (1 ligne par actif x période) -> CSV ;
  - sélectionne les périodes portant au moins un bon fit (R² > seuil) ;
  - rassemble les courbes retenues pour les figures ;
  - calcule les récapitulatifs de redondance et les résumés de fenêtres.
"""
import numpy as np
import pandas as pd

import config
from periods import asset_runs, strong_rise

R2_COLS = [f"R2_{m}" for m in config.SIS_MODELS]


def crisis_table(windows, period_data, ctx, crisis_map, csv_path=None):
    """Tableau 1 ligne par (actif, période) en crise ; R² par méthode -> CSV.

    Parameters
    ----------
    windows : list of (int, int)
        Fenêtres globales (intervals_chrono).
    period_data : dict
        ``{(pa, pb): (cache, fits)}`` issu de :func:`sis.fit_periods`.
    ctx : SimpleNamespace
        Contexte (asset_names, all_days, E_daily_all).
    crisis_map : ndarray of bool
        Carte de crise par actif (|C|).
    csv_path : str or Path or None
        Si fourni, écrit le tableau en CSV.

    Returns
    -------
    DataFrame
        Trié par période puis R² max décroissant.
    """
    rows = []
    for (pa, pb) in windows:
        cache, fits = period_data[(pa, pb)]
        for gi in cache:
            win = next(((a, b) for a, b in asset_runs(crisis_map[gi]) if b >= pa and a <= pb), None)
            if win is None:
                continue
            res = strong_rise(ctx.E_daily_all[gi, win[0]:win[1] + 1])
            if res is None:
                continue
            A0, B0 = win[0] + res[2], win[0] + res[3]
            row = {"actif": ctx.asset_names[gi], "debut": ctx.all_days[A0],
                   "fin": ctx.all_days[B0], "n_jours": B0 - A0 + 1,
                   "periode": f"{ctx.all_days[pa]}->{ctx.all_days[pb]}"}
            for m in config.SIS_MODELS:
                sl = fits[m][1].get(gi)
                row[f"R2_{m}"] = round(sl["r2"], 3) if sl is not None else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df = (df.assign(_r2max=df[R2_COLS].max(axis=1))
          .sort_values(["periode", "_r2max"], ascending=[True, False])
          .drop(columns="_r2max"))
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    return df


def select_periods(windows, period_data, r2_seuil=config.R2_SEUIL):
    """Périodes portant au moins un fit R² > ``r2_seuil`` (toutes méthodes).

    Returns
    -------
    list of (int, int)
    """
    return [(pa, pb) for (pa, pb) in windows
            if any(sl["r2"] > r2_seuil
                   for m in config.SIS_MODELS
                   for sl in period_data[(pa, pb)][1][m][1].values())]


def collect_by_method(period_data, selection, r2_seuil=config.R2_SEUIL):
    """Rassemble les courbes (R² > seuil) par méthode pour les figures « 3 vues ».

    Parameters
    ----------
    period_data : dict
        Fits par période.
    selection : list of (int, int)
        Périodes retenues.
    r2_seuil : float
        Seuil de sélection des fits.

    Returns
    -------
    dict
        ``{methode: dict(Es, Xs, E_all, X_all, r2s)}``.
    """
    out = {}
    for m in config.SIS_MODELS:
        Es, Xs, E_all, X_all, r2s = [], [], [], [], []
        for (pa, pb) in selection:
            fits = period_data[(pa, pb)][1]
            for gi, sl in fits[m][1].items():
                if sl["r2"] > r2_seuil:
                    E_i, x_traj = fits[m][0][gi]
                    ok = ~np.isnan(E_i)
                    Es.append(E_i); Xs.append(x_traj)
                    E_all.append(E_i[ok]); X_all.append(x_traj[ok]); r2s.append(sl["r2"])
        out[m] = dict(Es=Es, Xs=Xs, E_all=E_all, X_all=X_all, r2s=r2s)
    return out


def peak_table(peak_rises_list, peak_pd, ctx, r2_seuil=config.R2_SEUIL, csv_path=None):
    """Tableau des montées de pics (meilleure méthode par actif) + courbes retenues.

    Parameters
    ----------
    peak_rises_list : list of (int, int)
        Montées de pics (creux -> sommet).
    peak_pd : dict
        Fits SIS par montée.
    ctx : SimpleNamespace
        Contexte (asset_names, all_days).
    r2_seuil : float
        Seuil pour retenir une courbe.
    csv_path : str or Path or None
        Si fourni, écrit le tableau en CSV.

    Returns
    -------
    (DataFrame, dict)
        Le tableau trié par R² max et les courbes retenues
        ``dict(Es, Xs, E_all, X_all, r2s)``.
    """
    records = []
    curves = dict(Es=[], Xs=[], E_all=[], X_all=[], r2s=[])
    for (pa, pb) in peak_rises_list:
        cache, fits = peak_pd[(pa, pb)]
        for gi in cache:
            best_m, best_r2 = None, -1.0
            for m in config.SIS_MODELS:
                sl = fits[m][1].get(gi)
                if sl is not None and sl["r2"] > best_r2:
                    best_m, best_r2 = m, sl["r2"]
            if best_m is None:
                continue
            rec = {"actif": ctx.asset_names[gi],
                   "montee": f"{ctx.all_days[pa]}->{ctx.all_days[pb]}",
                   "sommet": str(ctx.all_days[pb]), "meilleure_methode": best_m}
            for m in config.SIS_MODELS:
                sl = fits[m][1].get(gi)
                rec[f"R2_{m}"] = round(sl["r2"], 3) if sl is not None else np.nan
            rec["R2_max"] = round(best_r2, 3)
            records.append(rec)
            if best_r2 > r2_seuil:
                E_i, x_traj = fits[best_m][0][gi]
                ok = ~np.isnan(E_i)
                curves["Es"].append(E_i); curves["Xs"].append(x_traj)
                curves["E_all"].append(E_i[ok]); curves["X_all"].append(x_traj[ok])
                curves["r2s"].append(best_r2)
    df = pd.DataFrame(records).sort_values("R2_max", ascending=False)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    return df, curves


def redundancy_recap(period_data, peak_pd, ctx, r2_seuil=config.R2_SEUIL):
    """Redondance (fits R²>seuil / actifs distincts) par méthode : sans | avec pics.

    Returns
    -------
    DataFrame
        Index = méthodes ; colonnes ``fits_sans, actifs_sans, redond_sans,
        fits_avec, actifs_avec, redond_avec``.
    """
    def _rows(pdict):
        rows = []
        for (pa, pb), (cache, fits) in pdict.items():
            for m in config.SIS_MODELS:
                for gi, sl in fits[m][1].items():
                    if sl["r2"] > r2_seuil:
                        rows.append((m, ctx.asset_names[gi]))
        return pd.DataFrame(rows, columns=["methode", "actif"])

    S, A = _rows(period_data), _rows(peak_pd)

    def _stat(df, m):
        g = df[df.methode == m]
        nu = g.actif.nunique()
        return len(g), nu, (len(g) / nu if nu else 0.0)

    out = {}
    for m in config.SIS_MODELS:
        fs, us, rs = _stat(S, m)
        fa, ua, ra = _stat(A, m)
        out[m] = dict(fits_sans=fs, actifs_sans=us, redond_sans=round(rs, 2),
                      fits_avec=fa, actifs_avec=ua, redond_avec=round(ra, 2))
    return pd.DataFrame(out).T


# ── Résumés de fenêtres pour la figure de comparaison (cell 27) ───────────────
def _r2max(fits, gi):
    return max((fits[m][1][gi]["r2"] for m in config.SIS_MODELS if gi in fits[m][1]),
               default=0.0)


def window_summaries(windows, pdict, all_days, r2_seuil=config.R2_SEUIL):
    """Pour chaque fenêtre : actifs en crise + nb de fits R²>seuil (les 2 boucles)."""
    out = []
    for (pa, pb) in windows:
        cache, fits = pdict[(pa, pb)]
        n_sup = sum(_r2max(fits, gi) > r2_seuil for gi in cache)
        out.append(dict(a=pd.Timestamp(all_days[pa]), b=pd.Timestamp(all_days[pb]),
                        cache=set(cache), n_crise=len(cache), n_sup=int(n_sup)))
    return sorted(out, key=lambda d: d["a"])


def retain(W, retenue_only=True):
    """Filtre les fenêtres retenues (>= 1 fit R²>seuil) si ``retenue_only``."""
    return [w for w in W if w["n_sup"] > 0] if retenue_only else W


def totals(W):
    """(nb actifs distincts en crise, nb total de fits R²>seuil) sur une liste de fenêtres."""
    crise = set().union(*[w["cache"] for w in W]) if W else set()
    return len(crise), sum(w["n_sup"] for w in W)
