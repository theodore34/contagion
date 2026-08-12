"""Orchestration : enchaîne données → périodes → SIS → sélection.

Couche fine et sans tracé, partagée par tous les scripts ``run_*.py``. Chaque
fonction renvoie des objets déjà mis en cache par les modules sous-jacents, donc
les appeler plusieurs fois (depuis des scripts différents) ne recalcule rien.
"""
import numpy as np
from scipy.stats import linregress

import config
import periods
import selection
import sis
from cache import disk_cache, sig_of
from networks import build_filtered_corr, fit_var, prepare_for_sis


def load_periods(ctx):
    """Boucle 1 (fenêtres globales) + boucle 2 (carte de crise par actif, |C|).

    Returns
    -------
    dict
        ``intervals_chrono, mc_smooth, crisis_map``.
    """
    intervals_chrono, mc_smooth = periods.detect_global_windows(ctx.mc_all, ctx.mean_mc)
    crisis_map = periods.compute_crisis_map(ctx.E_daily_all)
    return dict(intervals_chrono=intervals_chrono, mc_smooth=mc_smooth, crisis_map=crisis_map)


def fit_main(ctx, intervals_chrono, crisis_map):
    """Fits SIS sur les fenêtres croissantes (|C|) + périodes sélectionnées.

    Returns
    -------
    dict
        ``period_data`` (cache) et ``selected`` (périodes avec >=1 fit R²>seuil).
    """
    fc = sis.make_fit_context(ctx, crisis_map)
    sig = sis.sig_period_data(ctx.N, intervals_chrono, crisis_map)
    period_data = sis.fit_periods("period_data", sig, intervals_chrono, fc)
    selected = selection.select_periods(intervals_chrono, period_data)
    return dict(period_data=period_data, selected=selected)


def fit_peaks(ctx, crisis_map):
    """Détection par pics + montées + fits SIS sur les montées.

    Returns
    -------
    dict
        ``peak`` (sortie de detect_peaks), ``peak_rises``, ``peak_pd`` (cache).
    """
    peak = periods.detect_peaks(ctx.mc_all, ctx.mean_mc)
    pr = periods.peak_rises(peak["peak_intervals"], peak["mc_s"])
    fc = sis.make_fit_context(ctx, crisis_map)
    sig = sis.sig_peak_period_data(pr, crisis_map)
    peak_pd = sis.fit_periods("peak_period_data", sig, pr, fc)
    return dict(peak=peak, peak_rises=pr, peak_pd=peak_pd)


def fit_signed(ctx):
    """Pipeline « sans valeur absolue » : détection signée + fits SIS.

    Returns
    -------
    dict
        ``E_signed, mc_signed, mc_signed_smooth, intervals_signed,
        crisis_map_signed, period_data_sig, selected_sig``.
    """
    E_signed = periods.compute_E_signed(ctx.data, ctx.N)
    mc_signed, mc_signed_smooth, intervals_signed, cm_signed = \
        periods.detect_global_windows_signed(E_signed)
    fc = sis.make_fit_context(ctx, cm_signed, E_daily=E_signed)
    sig = sis.sig_period_data_signed(intervals_signed, cm_signed)
    period_data_sig = sis.fit_periods("period_data_sig", sig, intervals_signed, fc)
    selected_sig = selection.select_periods(intervals_signed, period_data_sig)
    return dict(E_signed=E_signed, mc_signed=mc_signed, mc_signed_smooth=mc_signed_smooth,
                intervals_signed=intervals_signed, crisis_map_signed=cm_signed,
                period_data_sig=period_data_sig, selected_sig=selected_sig)


def spectral(ctx):
    """Diagnostic spectral λ_max/⟨λ⟩ + pics + périodes 'mix' (mis en cache)."""
    diag = periods.spectral_diagnostics(ctx.data, ctx.N)
    return dict(diag=diag, lam_dates=periods.lambda_peaks(diag),
                mix_periods=periods.mix_periods(diag))


# ── Expériences ───────────────────────────────────────────────────────────────
def _corr_thr_fits(q, ctx, intervals_chrono, period_data):
    """Fits 'Corr thr' au quantile q sur tous les actifs en crise (toutes périodes).

    Réutilise les conditions initiales déjà en cache (``period_data[...][0]``),
    indépendantes de q ; seule la matrice 'Corr thr' dépend de q.
    """
    A_q = build_filtered_corr(ctx.C_full, q, ctx.mask_off)
    Es, Xs, E_all, X_all, r2s, keys = [], [], [], [], [], []
    for (pa, pb) in intervals_chrono:
        cache = period_data[(pa, pb)][0]
        for gi in cache:
            E_i = cache[gi]["E_i"]
            xt = sis.integrate_fast(cache[gi], gi, A_q)
            ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            f = linregress(E_i[ok], xt[ok])
            Es.append(E_i); Xs.append(xt)
            E_all.append(E_i[ok]); X_all.append(xt[ok])
            r2s.append(f.rvalue ** 2); keys.append((pa, pb, gi))
    return Es, Xs, E_all, X_all, r2s, keys


def run_qsweep(ctx, intervals_chrono, crisis_map, period_data):
    """Balayage du seuil q de 'Corr thr' (parallélisé, mis en cache).

    Returns
    -------
    dict
        ``{q: (Es, Xs, E_all, X_all, r2s, keys)}``.
    """
    import os
    from joblib import Parallel, delayed

    sig = sig_of(config.Q_SWEEP, intervals_chrono, crisis_map,
                 config.B_FIT, config.R_FIT, config.T_LONG, config.TOL_EQ, "fast")

    def _compute():
        res = Parallel(n_jobs=min(len(config.Q_SWEEP), config.N_JOBS))(
            delayed(_corr_thr_fits)(q, ctx, intervals_chrono, period_data)
            for q in config.Q_SWEEP)
        return {round(float(q), 2): r for q, r in zip(config.Q_SWEEP, res)}

    return disk_cache("qsweep_corrthr", sig, _compute)


def _br_counts(B, R, caches, A_sis, r2_seuil):
    """Pour un couple (B, R) : par méthode, nb de fits et d'actifs distincts R²>seuil.

    Réutilise les conditions initiales déjà en cache (``caches``), qui ne dépendent
    pas de (B, R) ; seule l'intégration SIS en dépend.

    Parameters
    ----------
    B, R : float
        Taux de récupération et d'infection.
    caches : list of dict
        Une entrée ``cache`` (gi -> dict(E_i, x0, n_days)) par fenêtre.
    A_sis : dict
        Matrices de contagion par méthode.
    r2_seuil : float
        Seuil de sélection des fits.

    Returns
    -------
    dict
        ``{methode: dict(n_fits, n_assets, med_r2_all, mean_r2_all, med_r2_sel,
        agg_r2_all, agg_r2_sel)}``. ``med_*``/``mean_*`` sont des R² médians/moyens par
        courbe ; ``agg_*`` est le R² de la régression agrégée sur le nuage poolé E↔x
        (``_sel`` = fits R²>seuil, ``_all`` = tous ; ``nan`` si vide).
    """
    fits = {m: 0 for m in config.SIS_MODELS}
    assets = {m: set() for m in config.SIS_MODELS}
    r2all = {m: [] for m in config.SIS_MODELS}
    Es = {m: [] for m in config.SIS_MODELS}
    Xs = {m: [] for m in config.SIS_MODELS}
    Esel = {m: [] for m in config.SIS_MODELS}
    Xsel = {m: [] for m in config.SIS_MODELS}
    for cache in caches:
        for m in config.SIS_MODELS:
            A = A_sis[m]
            for gi, d in cache.items():
                E_i = d["E_i"]
                ok = ~np.isnan(E_i)
                if ok.sum() < 3:
                    continue
                xt = sis.integrate_xscan(d, gi, A, B=B, R=R)   # intégrateur léger (balayage)
                r2 = linregress(E_i[ok], xt[ok]).rvalue ** 2
                r2all[m].append(r2)
                Es[m].append(E_i[ok])
                Xs[m].append(xt[ok])
                if r2 > r2_seuil:
                    fits[m] += 1
                    assets[m].add(gi)
                    Esel[m].append(E_i[ok])
                    Xsel[m].append(xt[ok])

    def _agg(Ed, Xd):
        if not Ed:
            return np.nan
        return float(linregress(np.concatenate(Ed), np.concatenate(Xd)).rvalue ** 2)

    out = {}
    for m in config.SIS_MODELS:
        arr = np.array(r2all[m], float)
        sel = arr[arr > r2_seuil]
        out[m] = dict(n_fits=fits[m], n_assets=len(assets[m]),
                      med_r2_all=float(np.median(arr)) if arr.size else np.nan,
                      mean_r2_all=float(arr.mean()) if arr.size else np.nan,
                      med_r2_sel=float(np.median(sel)) if sel.size else np.nan,
                      agg_r2_all=_agg(Es[m], Xs[m]), agg_r2_sel=_agg(Esel[m], Xsel[m]))
    return out


def run_br_scan(ctx, windows, crisis_map, period_data, B_grid, R_grid):
    """Balaye la grille (B, R) -> actifs/fits retenus par méthode (parallélisé, caché).

    Parameters
    ----------
    ctx : SimpleNamespace
        Contexte (pour ``A_sis``).
    windows : list of (int, int)
        Fenêtres sur lesquelles compter (typiquement ``intervals_chrono``).
    crisis_map : ndarray of bool
        Carte de crise (n'entre que dans la signature de cache).
    period_data : dict
        Fits déjà calculés (on n'en réutilise que les conditions initiales).
    B_grid, R_grid : array-like
        Valeurs de B et de R balayées.

    Returns
    -------
    dict
        ``{(B, R): {methode: dict(n_fits, n_assets, med_r2_all, mean_r2_all, med_r2_sel)}}``.
    """
    import os
    from joblib import Parallel, delayed

    caches = [period_data[(pa, pb)][0] for (pa, pb) in windows]
    pairs = [(round(float(B), 2), round(float(R), 2)) for B in B_grid for R in R_grid]
    sig = sig_of(np.asarray(B_grid, float), np.asarray(R_grid, float), windows, crisis_map,
                 config.CORR_THRESHOLD, config.T_LONG, config.TOL_EQ, config.R2_SEUIL, "br_fast_r2")

    def _compute():
        res = Parallel(n_jobs=min(len(pairs), config.N_JOBS))(
            delayed(_br_counts)(B, R, caches, ctx.A_sis, config.R2_SEUIL) for (B, R) in pairs)
        return {p: r for p, r in zip(pairs, res)}

    return disk_cache("br_scan", sig, _compute)


def _r2_for_A(A_model, ctx, selected, period_data):
    """R² agrégés (x_i SIS vs E_i) sur 'selected' pour une matrice de contagion."""
    r2s, Ea, Xa = [], [], []
    for (pa, pb) in selected:
        cache = period_data[(pa, pb)][0]
        for gi in cache:
            E_i = cache[gi]["E_i"]
            ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            xt = sis.integrate_xscan(cache[gi], gi, A_model)
            r2s.append(linregress(E_i[ok], xt[ok]).rvalue ** 2)
            Ea.append(E_i[ok]); Xa.append(xt[ok])
    r2s = np.array(r2s)
    agg = linregress(np.concatenate(Ea), np.concatenate(Xa)).rvalue ** 2 if Ea else np.nan
    return dict(n=len(r2s), n_sup=int((r2s > config.R2_SEUIL).sum()),
                R2_moyen=float(np.nanmean(r2s)), R2_agrege=float(agg))


def run_lag_scan(ctx, selected, period_data):
    """Balayage des lags VAR -> qualité du fit SIS (mis en cache).

    Returns
    -------
    DataFrame
        Index = lag VAR ; colonnes ``n, n_sup, R2_moyen, R2_agrege``.
    """
    import pandas as pd

    lags = config.LAG_SCAN

    def _scan():
        return {d: _r2_for_A(prepare_for_sis(np.abs(fit_var(ctx.data.values, d).T)),
                             ctx, selected, period_data) for d in lags}

    sig = (f"lags{lags[0]}-{lags[-1]}_xscan_sel{len(selected)}_T{config.T_LONG}"
           f"_B{config.B_FIT}_R{config.R_FIT}_tol{config.TOL_EQ}_{sig_of(selected)}")
    lag_res = disk_cache("var_lag_scan", sig, _scan)
    ldf = pd.DataFrame(lag_res).T
    ldf.index.name = "lag_VAR"
    return ldf
