"""Dynamique SIS — la brique « un qui résout la dynamique SIS en prenant la matrice ».

Le modèle SIS borné

    dx_i/dt = -B x_i + R (1 - x_i) Σ_j A_ij x_j ,   x_i ∈ [0, 1]

est intégré à partir d'une condition initiale x0 (profil E_i au premier jour de
la montée), pour une matrice de contagion ``A`` quelconque (PMFG, VAR, Corr thr).
La trajectoire x_i(t) est recadrée sur le temps de convergence T_conv vers
l'équilibre x*, puis comparée au profil empirique E_i(t) par régression -> R².

Trois intégrateurs de fidélité décroissante (tous équivalents au point fixe,
seul le coût change) :
  - :func:`integrate`        : double intégration, grille 5000 pts (référence)
  - :func:`integrate_fast`   : simple intégration, grille 5000 pts (~2x plus rapide)
  - :func:`integrate_xscan`  : tolérances relâchées, grille 400 pts (balayages)

`fit_periods` calcule et met en cache, pour une liste de fenêtres, les fits SIS
de chaque méthode.
"""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress

import config
from cache import disk_cache, sig_of
from periods import asset_runs, strong_rise


def _sis_params_tag():
    """Suffixe lisible des paramètres SIS communs aux signatures de cache."""
    return (f"thr{config.CORR_THRESHOLD}_T{config.T_LONG}_B{config.B_FIT}"
            f"_R{config.R_FIT}_tol{config.TOL_EQ}"
            f"_sr{config.SR_SMOOTH}-{config.SR_MIN_LEN}-{config.SR_MIN_GAIN}")


def sig_period_data(N, windows, crisis_map):
    """Signature du cache 'period_data' (fenêtres croissantes, |C|)."""
    return f"N{N}_p{len(windows)}_{_sis_params_tag()}_{sig_of(windows, crisis_map)}"


def sig_period_data_signed(windows_signed, crisis_map_signed):
    """Signature du cache 'period_data_sig' (détection signée, sans np.abs)."""
    return (f"SIG_p{len(windows_signed)}_{_sis_params_tag()}"
            f"_{sig_of(windows_signed, crisis_map_signed)}")


def sig_peak_period_data(peak_rises_list, crisis_map):
    """Signature du cache 'peak_period_data' (montées des pics)."""
    return sig_of(peak_rises_list, crisis_map, config.CORR_THRESHOLD, config.T_LONG,
                  config.B_FIT, config.R_FIT, config.TOL_EQ,
                  config.SR_SMOOTH, config.SR_MIN_LEN, config.SR_MIN_GAIN)


def ode_sis_bounded(t, x, A, B, R):
    """Champ de vitesse du SIS borné : état clippé dans [0, 1].

    Le clipping rend la dynamique stable même pour une condition initiale signée.

    Parameters
    ----------
    t : float
        Temps (ignoré, système autonome).
    x : ndarray
        État courant.
    A : ndarray, shape (N, N)
        Matrice de contagion.
    B, R : float
        Taux de récupération et d'infection.

    Returns
    -------
    ndarray
        dx/dt.
    """
    xc = np.clip(x, 0.0, 1.0)
    return -B * xc + R * (1.0 - xc) * (A @ xc)


def _solve(x0, A, t_span, t_eval=None, rtol=1e-6, atol=1e-9, B=None, R=None):
    """Raccourci ``solve_ivp`` (LSODA) ; B / R par défaut = ceux de config."""
    B = config.B_FIT if B is None else B
    R = config.R_FIT if R is None else R
    return solve_ivp(ode_sis_bounded, t_span, x0, args=(A, B, R),
                     t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol)


def integrate(cache_gi, gi, A, B=None, R=None):
    """Trajectoire x_i(t) (référence) recadrée sur le temps de convergence.

    x* est obtenu par une 1re intégration à ``T_LONG`` ; le temps de convergence
    T_conv est le 1er instant où x_i atteint ``(1 - TOL_EQ) x*``.

    Parameters
    ----------
    cache_gi : dict
        Entrée ``cache[gi]`` : ``x0`` (condition initiale) et ``n_days``.
    gi : int
        Indice global de l'actif suivi.
    A : ndarray, shape (N, N)
        Matrice de contagion.
    B, R : float or None
        Taux de récupération / d'infection ; ``None`` -> valeurs de config.

    Returns
    -------
    ndarray, shape (n_days,)
        Trajectoire x_i recadrée et clippée dans ]0, 1[.
    """
    x0, n = cache_gi["x0"], cache_gi["n_days"]
    x_eq = _solve(x0, A, (0, config.T_LONG), B=B, R=R).y[gi, -1]
    tp = np.linspace(1e-3, config.T_LONG, 5000)
    sp = _solve(x0, A, (0, config.T_LONG), t_eval=tp, B=B, R=R)
    above = np.where(sp.y[gi] >= (1 - config.TOL_EQ) * x_eq)[0]
    Tc = float(tp[above[0]]) if len(above) else config.T_LONG
    s = _solve(x0, A, (0, Tc), t_eval=np.linspace(0, Tc, n), B=B, R=R)
    return np.clip(s.y[gi], 1e-8, 1 - 1e-8)


def integrate_fast(cache_gi, gi, A, B=None, R=None):
    """Comme :func:`integrate` mais sans la 1re intégration (x* = dernier point).

    Résultat identique au point fixe, ~2x plus rapide ; utilisé pour les balayages
    (seuil q et taux B/R). ``B`` / ``R`` à ``None`` -> valeurs de config.
    """
    x0, n = cache_gi["x0"], cache_gi["n_days"]
    tp = np.linspace(1e-3, config.T_LONG, 5000)
    sp = _solve(x0, A, (0, config.T_LONG), t_eval=tp, B=B, R=R)
    x_eq = sp.y[gi, -1]
    above = np.where(sp.y[gi] >= (1 - config.TOL_EQ) * x_eq)[0]
    Tc = float(tp[above[0]]) if len(above) else config.T_LONG
    s = _solve(x0, A, (0, Tc), t_eval=np.linspace(0, Tc, n), B=B, R=R)
    return np.clip(s.y[gi], 1e-8, 1 - 1e-8)


def integrate_xscan(cache_gi, gi, A, B=None, R=None):
    """Intégrateur allégé (rtol 1e-4, grille 400 pts) pour le balayage des lags VAR."""
    x0, n = cache_gi["x0"], cache_gi["n_days"]
    kw = dict(rtol=1e-4, atol=1e-7, B=B, R=R)
    x_eq = _solve(x0, A, (0, config.T_LONG), **kw).y[gi, -1]
    tp = np.linspace(1e-3, config.T_LONG, 400)
    sp = _solve(x0, A, (0, config.T_LONG), t_eval=tp, **kw)
    above = np.where(sp.y[gi] >= (1 - config.TOL_EQ) * x_eq)[0]
    Tc = float(tp[above[0]]) if len(above) else config.T_LONG
    s = _solve(x0, A, (0, Tc), t_eval=np.linspace(0, Tc, n), **kw)
    return np.clip(s.y[gi], 1e-8, 1 - 1e-8)


# ── Contexte de fit (données + carte de crise + matrices) ─────────────────────
def make_fit_context(ctx, crisis_map, E_daily=None):
    """Empaquette ce dont :func:`fits_for` a besoin (réutilisable pour le signé).

    Parameters
    ----------
    ctx : SimpleNamespace
        Sortie de ``data.build_context``.
    crisis_map : ndarray of bool
        Carte de crise par actif (|C| ou signée).
    E_daily : ndarray or None
        Profils E_i journaliers ; ``None`` -> ``ctx.E_daily_all`` (cas |C|).

    Returns
    -------
    SimpleNamespace
        Champs ``data, all_days, N, crisis_map, E_daily, A_sis``.
    """
    return SimpleNamespace(data=ctx.data, all_days=ctx.all_days, N=ctx.N,
                           crisis_map=crisis_map,
                           E_daily=ctx.E_daily_all if E_daily is None else E_daily,
                           A_sis=ctx.A_sis)


def cache_signed(pa, pb, fc):
    """Pour chaque actif en crise sur (pa, pb) : E_i mesuré + condition initiale x0.

    L'actif n'est retenu que s'il présente une montée forte (``strong_rise``)
    dans sa fenêtre de crise chevauchant (pa, pb).

    Parameters
    ----------
    pa, pb : int
        Bornes (indices de jours) de la période globale.
    fc : SimpleNamespace
        Contexte de fit (:func:`make_fit_context`).

    Returns
    -------
    dict
        ``{gi: dict(E_i, x0, n_days)}`` pour les actifs retenus.
    """
    rw = {}
    for gi in range(fc.N):
        win = next(((a, b) for a, b in asset_runs(fc.crisis_map[gi]) if b >= pa and a <= pb), None)
        if win is None:
            continue
        res = strong_rise(fc.E_daily[gi, win[0]:win[1] + 1])
        if res is not None:
            rw[gi] = (win[0] + res[2], win[0] + res[3])

    cache = {}
    for gi0, (A0, B0) in rw.items():
        days_p = list(fc.all_days[A0:B0 + 1])
        dp = fc.data[np.isin(fc.data.index.date, days_p)].copy()
        dp["day"] = dp.index.date
        days_s = sorted(dp["day"].unique())
        n_days = len(days_s)
        Es = np.full((fc.N, n_days), np.nan)
        for k, d in enumerate(days_s):
            chunk = dp[dp["day"] == d].drop(columns=["day"]).values
            if len(chunk) < 3:
                continue
            C = np.nan_to_num(np.corrcoef(chunk.T), nan=0.0)
            np.fill_diagonal(C, 0)
            Es[:, k] = C.sum(axis=1) / (fc.N - 1)
        valid = np.where(np.any(~np.isnan(Es), axis=0))[0]
        if len(valid) == 0:
            continue
        cache[gi0] = dict(E_i=Es[gi0], x0=Es[:, valid[0]], n_days=n_days)
    return cache


def fits_for(pa, pb, fc):
    """Fits SIS de toutes les méthodes sur la période (pa, pb).

    Parameters
    ----------
    pa, pb : int
        Bornes de la période globale.
    fc : SimpleNamespace
        Contexte de fit.

    Returns
    -------
    (dict, dict)
        ``cache`` (sortie de :func:`cache_signed`) et ``fits`` :
        ``fits[m] = (pad, sl)`` où ``pad[gi] = (E_i, x_traj)`` et
        ``sl[gi] = dict(slope, r2, n)``.
    """
    cache = cache_signed(pa, pb, fc)
    fits = {}
    for m in config.SIS_MODELS:
        A_m, pad, sl = fc.A_sis[m], {}, {}
        for gi in cache:
            E_i = cache[gi]["E_i"]
            xt = integrate(cache[gi], gi, A_m)
            pad[gi] = (E_i, xt)
            ok = ~np.isnan(E_i)
            if ok.sum() >= 3:
                f = linregress(E_i[ok], xt[ok])
                sl[gi] = dict(slope=f.slope, r2=f.rvalue ** 2, n=int(ok.sum()))
        fits[m] = (pad, sl)
    return cache, fits


def fit_periods(name, sig, windows, fc):
    """Calcule (et met en cache) ``{(pa, pb): fits_for(pa, pb)}`` sur des fenêtres.

    Parameters
    ----------
    name : str
        Préfixe de cache ('period_data', 'peak_period_data', 'period_data_sig').
    sig : str
        Signature de cache (doit refléter fenêtres + carte de crise + params SIS).
    windows : list of (int, int)
        Périodes à ajuster.
    fc : SimpleNamespace
        Contexte de fit.

    Returns
    -------
    dict
        ``{(pa, pb): (cache, fits)}``.
    """
    return disk_cache(name, sig,
                      lambda: {(pa, pb): fits_for(pa, pb, fc) for (pa, pb) in windows})
