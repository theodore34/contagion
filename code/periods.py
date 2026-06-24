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
from cache import disk_cache


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
