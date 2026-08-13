"""Détection des périodes de crise — la brique « un pour les périodes ».

Trois familles de périodes sont produites à partir de la corrélation moyenne
journalière ⟨|C_ij|⟩ et des profils par actif E_i :

  1. fenêtres CROISSANTES de ⟨|C|⟩          -> :func:`detect_global_windows`
  2. montées des PICS de ⟨|C|⟩              -> :func:`detect_peaks` + :func:`peak_rises`
  3. crise PAR ACTIF (E_i croissant)        -> :func:`compute_crisis_map` (mis en cache)

Une variante « signée » (sans valeur absolue) est fournie pour l'expérience de
détection sans ``np.abs`` (:func:`compute_E_signed`).

Le diagnostic spectral λ_max/⟨λ⟩ (:func:`spectral_diagnostics`, mis en cache)
sert de point de comparaison « mode de marché ».
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

import config
from config import disk_cache


# ── Outils génériques de détection ────────────────────────────────────────────
def increasing_intervals(y, smooth=config.SMOOTH, min_len=config.MIN_LEN,
                         min_gain=0.0, merge_gap=config.MERGE_GAP, refine=config.REFINE):
    """Intervalles (début, fin) où ``y`` lissé croît ; triés par gain décroissant.

    Parameters
    ----------
    y : array-like
        Série à analyser.
    smooth : int
        Largeur de la moyenne glissante appliquée avant la détection.
    min_len : int
        Longueur minimale d'un intervalle conservé.
    min_gain : float
        Gain (croissance lissée) minimal exigé sur l'intervalle.
    merge_gap : int
        Fusionne deux intervalles séparés par <= ``merge_gap`` points décroissants.
    refine : int
        Demi-fenêtre d'ajustement fin des bords pour maximiser l'amplitude.

    Returns
    -------
    (list of (int, int), ndarray)
        Les intervalles triés par gain décroissant, et la série lissée ``ys``.
    """
    y = np.asarray(y, float)
    ys = uniform_filter1d(y, size=smooth, mode="nearest")
    inc = np.gradient(ys) > 0
    raw, s = [], None
    for k, v in enumerate(inc):
        if v and s is None:
            s = k
        elif not v and s is not None:
            raw.append((s, k - 1)); s = None
    if s is not None:
        raw.append((s, len(inc) - 1))
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = [(a, b) for a, b in merged
           if (b - a + 1) >= min_len and (ys[b] - ys[a]) >= min_gain]
    if refine > 0:
        n, adj = len(y), []
        for a0, b0 in out:
            ba, bb, bamp = a0, b0, y[b0] - y[a0]
            for a in range(max(0, a0 - refine), min(n, a0 + refine + 1)):
                for b in range(max(0, b0 - refine), min(n, b0 + refine + 1)):
                    if (b - a + 1) >= min_len and y[b] - y[a] > bamp:
                        ba, bb, bamp = a, b, y[b] - y[a]
            adj.append((ba, bb))
        out = adj
    out.sort(key=lambda ab: ys[ab[1]] - ys[ab[0]], reverse=True)
    return out, ys


def asset_runs(row):
    """Plages contiguës ``True`` d'un booléen -> liste de (début, fin) inclusifs.

    Parameters
    ----------
    row : array-like of bool
        Masque temporel.

    Returns
    -------
    list of (int, int)
        Les plages contiguës de valeurs vraies.
    """
    runs, s = [], None
    for k, v in enumerate(row):
        if v and s is None:
            s = k
        elif not v and s is not None:
            runs.append((s, k - 1)); s = None
    if s is not None:
        runs.append((s, len(row) - 1))
    return runs


def strong_rise(y, smooth=config.SR_SMOOTH, min_len=config.SR_MIN_LEN,
                min_gain=config.SR_MIN_GAIN):
    """Plus longue sous-fenêtre de forte montée (gain lissé >= ``min_gain``).

    Parameters
    ----------
    y : array-like
        Profil E_i sur la fenêtre de crise de l'actif.
    smooth, min_len, min_gain : voir config.SR_*

    Returns
    -------
    tuple or None
        ``(longueur, gain, start, end)`` de la meilleure sous-fenêtre, ou None.
    """
    ys, best, m = uniform_filter1d(y, size=smooth, mode="nearest"), None, len(y)
    for s in range(m):
        for e in range(s + min_len - 1, m):
            gain = ys[e] - ys[s]
            if gain >= min_gain:
                cand = (e - s + 1, gain, s, e)
                if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand
    return best


# ── Boucle 1 : fenêtres globales de crise ─────────────────────────────────────
def detect_global_windows(mc_all, mean_mc):
    """Fenêtres globales de crise : ⟨|C|⟩ croissant ET niveau > seuil de crise.

    Parameters
    ----------
    mc_all : ndarray
        ⟨|C_ij|⟩ journalier.
    mean_mc : float
        Moyenne globale de ⟨|C|⟩.

    Returns
    -------
    (list of (int, int), ndarray)
        Fenêtres en ordre chronologique, et la série lissée.
    """
    intervals_raw, mc_smooth = increasing_intervals(mc_all)
    chrono = sorted([(a, b) for a, b in intervals_raw
                     if np.nanmean(mc_all[a:b + 1]) > config.CRISIS_FACTOR * mean_mc],
                    key=lambda ab: ab[0])
    return chrono, mc_smooth


# ── Boucle 2 : crise par actif (E_i croissant) — mise en cache ────────────────
def _crisis_map_from(E):
    """Carte (N x jours) des plages où chaque E_i croît au-dessus de son seuil."""
    N = E.shape[0]
    cm = np.zeros((N, E.shape[1]), dtype=bool)
    for i in range(N):
        y = E[i]
        thr = config.ASSET_FACTOR * np.nanmean(y)
        for a, b in increasing_intervals(y)[0]:
            if np.nanmean(y[a:b + 1]) > thr:
                cm[i, a:b + 1] = True
    return cm


def compute_crisis_map(E_daily_all):
    """Carte de crise par actif (|C|), mise en cache sous les paramètres de détection.

    Returns
    -------
    ndarray of bool, shape (N, jours)
    """
    sig = (f"af{config.ASSET_FACTOR}_s{config.SMOOTH}_ml{config.MIN_LEN}"
           f"_mg{config.MERGE_GAP}_rf{config.REFINE}")
    return disk_cache("crisis_map", sig, lambda: _crisis_map_from(E_daily_all))


# ── Variante signée (expérience « sans valeur absolue ») ──────────────────────
def compute_E_signed(data, N):
    """E_i journalier SIGNÉ (somme des C_ij sans ``np.abs``), mis en cache.

    Parameters
    ----------
    data : DataFrame
        Log-rendements (index horodaté).
    N : int
        Nombre d'actifs.

    Returns
    -------
    ndarray, shape (N, jours)
    """
    def _compute():
        da = data.copy()
        da["day"] = da.index.date
        days = np.array(sorted(da["day"].unique()))
        E = np.full((N, len(days)), np.nan)
        for k, dd in enumerate(days):
            chunk = da[da["day"] == dd].drop(columns=["day"]).values
            if len(chunk) < 3:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.nan_to_num(np.corrcoef(chunk.T), nan=0.0)
            np.fill_diagonal(C, 0)
            E[:, k] = C.sum(axis=1) / (N - 1)        # SIGNÉ : pas de np.abs
        return E

    return disk_cache("E_daily_signed", f"N{N}T{len(data)}", _compute)


def detect_global_windows_signed(E_signed):
    """Fenêtres globales + carte de crise sur la corrélation SIGNÉE.

    Returns
    -------
    (mc_signed, mc_signed_smooth, intervals_signed, crisis_map_signed)
    """
    mc_signed = np.nanmean(E_signed, axis=0)         # = moyenne des C_ij signés
    mean_mc_signed = np.nanmean(mc_signed)
    iv_raw, mc_smooth = increasing_intervals(mc_signed)
    intervals = sorted([(a, b) for a, b in iv_raw
                        if np.nanmean(mc_signed[a:b + 1]) > config.CRISIS_FACTOR * mean_mc_signed],
                       key=lambda ab: ab[0])
    crisis_map_signed = _crisis_map_from(E_signed)
    return mc_signed, mc_smooth, intervals, crisis_map_signed


# ── Détection par les pics de ⟨|C|⟩ ───────────────────────────────────────────
def detect_peaks(mc_all, mean_mc):
    """Pics de ⟨|C|⟩ lissé et largeur de chaque pic -> intervalles de crise.

    Parameters
    ----------
    mc_all : ndarray
        ⟨|C_ij|⟩ journalier.
    mean_mc : float
        Moyenne globale (sert de hauteur minimale d'un pic).

    Returns
    -------
    dict
        ``mc_s`` (série lissée), ``peaks`` (indices des sommets),
        ``w_h`` / ``l_ips`` / ``r_ips`` (hauteur et bornes de largeur),
        ``peak_intervals`` (liste de (gauche, droite, sommet)).
    """
    mc_s = uniform_filter1d(np.nan_to_num(mc_all, nan=mean_mc),
                            size=config.SMOOTH, mode="nearest")
    prom = config.PK_PROM_FACTOR * np.nanstd(mc_all)
    peaks, _ = find_peaks(mc_s, height=mean_mc, prominence=prom, distance=config.PK_DISTANCE)
    _w, w_h, l_ips, r_ips = peak_widths(mc_s, peaks, rel_height=config.PK_RELHEIGHT)
    peak_intervals = sorted(
        [(int(np.floor(l)), int(np.ceil(r)), int(p)) for l, r, p in zip(l_ips, r_ips, peaks)],
        key=lambda t: t[0])
    return dict(mc_s=mc_s, peaks=peaks, w_h=w_h, l_ips=l_ips, r_ips=r_ips,
                peak_intervals=peak_intervals)


def peak_rises(peak_intervals, mc_s):
    """Montées des pics : du creux précédent jusqu'au sommet (>= MIN_LEN jours).

    Parameters
    ----------
    peak_intervals : list of (int, int, int)
        Sortie ``peak_intervals`` de :func:`detect_peaks`.
    mc_s : ndarray
        ⟨|C|⟩ lissé (pour localiser les creux).

    Returns
    -------
    list of (int, int)
        Montées (creux -> sommet), triées et dédupliquées.
    """
    rises, prev = [], 0
    for a, b, pk in peak_intervals:
        trough = prev + int(np.argmin(mc_s[prev:pk + 1])) if pk > prev else pk
        if pk - trough + 1 >= config.MIN_LEN:
            rises.append((trough, pk))
        prev = pk
    return sorted(set(rises))


# ── Diagnostic spectral λ_max/⟨λ⟩ (mode de marché) ────────────────────────────
def spectral_diagnostics(data, N):
    """λ_max/⟨λ⟩ et max|ρ|/⟨|ρ|⟩ en fenêtre glissante, mis en cache.

    Parameters
    ----------
    data : DataFrame
        Log-rendements.
    N : int
        Nombre d'actifs.

    Returns
    -------
    DataFrame
        Index daté ; colonnes ``ratio_corr, ratio_eig, mean_return,
        max_corr, mean_corr``.
    """
    import pandas as pd

    def _compute():
        a = data.values
        Tn, Nn = a.shape
        iu = np.triu_indices(Nn, k=1)
        rows, idx = [], []
        for end in range(config.WINDOW_SPEC, Tn + 1, config.STEP_SPEC):
            sub = a[end - config.WINDOW_SPEC:end]
            C = np.corrcoef(sub.T)
            off = np.abs(C[iu])
            ev = np.linalg.eigvalsh(C)
            rows.append((off.max() / off.mean(), ev.max() / ev.mean(),
                         sub.mean(), off.max(), off.mean()))
            idx.append(data.index[end - 1])
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                            columns=["ratio_corr", "ratio_eig", "mean_return",
                                     "max_corr", "mean_corr"])

    sig = f"N{N}T{len(data)}_w{config.WINDOW_SPEC}s{config.STEP_SPEC}_v2"
    return disk_cache("spectral_diag", sig, _compute)


def lambda_peaks(diag):
    """Dates des pics de λ_max/⟨λ⟩ lissé.

    Parameters
    ----------
    diag : DataFrame
        Sortie de :func:`spectral_diagnostics`.

    Returns
    -------
    DatetimeIndex
        Dates des sommets de λ_max/⟨λ⟩.
    """
    re = uniform_filter1d(diag["ratio_eig"].values, size=config.SPEC_SMOOTH, mode="nearest")
    pk, _ = find_peaks(re, prominence=config.SPEC_PROM_FACTOR * diag["ratio_eig"].std(),
                       distance=20)
    return diag.index[pk]


def mix_periods(diag):
    """Périodes « mix » : bosses du score z(λ_max/⟨λ⟩) + z(max|ρ|/⟨|ρ|⟩).

    Parameters
    ----------
    diag : DataFrame
        Sortie de :func:`spectral_diagnostics`.

    Returns
    -------
    list of (Timestamp, Timestamp)
        Bornes (gauche, droite) de chaque bosse du score combiné.
    """
    def _z(s):
        s = np.asarray(s, float)
        return (s - np.nanmean(s)) / np.nanstd(s)

    mix = uniform_filter1d(_z(diag["ratio_eig"]) + _z(diag["ratio_corr"]),
                           size=config.SPEC_SMOOTH, mode="nearest")
    mpk, _ = find_peaks(mix, prominence=0.8 * np.nanstd(mix), distance=20)
    _lw, _wh, li, ri = peak_widths(mix, mpk, rel_height=0.5)
    return sorted({(diag.index[int(np.floor(l))],
                    diag.index[min(int(np.ceil(r)), len(diag) - 1)])
                   for l, r in zip(li, ri)})


# ═══════════════════════════ Dynamique SIS (ex-sis.py) ═══════════════════════════

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
from config import disk_cache, sig_of


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
