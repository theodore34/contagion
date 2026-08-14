"""Crisis-period detection — the "one for the periods" building block.

Three families of periods are produced from the daily mean correlation ⟨|C_ij|⟩
and the per-asset profiles E_i:

  1. RISING windows of ⟨|C|⟩              -> :func:`detect_global_windows`
  2. RISES to the PEAKS of ⟨|C|⟩          -> :func:`detect_peaks` + :func:`peak_rises`
  3. PER-ASSET crisis (rising E_i)        -> :func:`compute_crisis_map` (cached)

A "signed" variant (without absolute value) is provided for the detection
experiment without ``np.abs`` (:func:`compute_E_signed`).

The spectral diagnostic λ_max/⟨λ⟩ (:func:`spectral_diagnostics`, cached) serves
as a "market mode" reference point.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

import config
from config import disk_cache


# ── Generic detection tools ───────────────────────────────────────────────────
def increasing_intervals(y, smooth=config.SMOOTH, min_len=config.MIN_LEN,
                         min_gain=0.0, merge_gap=config.MERGE_GAP, refine=config.REFINE):
    """Intervals (start, end) where smoothed ``y`` rises; sorted by decreasing gain.

    Parameters
    ----------
    y : array-like
        Series to analyse.
    smooth : int
        Width of the moving average applied before detection.
    min_len : int
        Minimum length of a kept interval.
    min_gain : float
        Minimum (smoothed) rise required over the interval.
    merge_gap : int
        Merge two intervals separated by <= ``merge_gap`` declining points.
    refine : int
        Half-window for fine edge adjustment maximising the amplitude.

    Returns
    -------
    (list of (int, int), ndarray)
        The intervals sorted by decreasing gain, and the smoothed series ``ys``.
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
    """Contiguous ``True`` runs of a boolean -> list of inclusive (start, end).

    Parameters
    ----------
    row : array-like of bool
        Temporal mask.

    Returns
    -------
    list of (int, int)
        The contiguous runs of true values.
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
    """Longest strong-rise sub-window (smoothed gain >= ``min_gain``).

    Parameters
    ----------
    y : array-like
        E_i profile over the asset's crisis window.
    smooth, min_len, min_gain : see config.SR_*

    Returns
    -------
    tuple or None
        ``(length, gain, start, end)`` of the best sub-window, or None.
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


# ── Loop 1: global crisis windows ─────────────────────────────────────────────
def detect_global_windows(mc_all, mean_mc):
    """Global crisis windows: rising ⟨|C|⟩ AND level > crisis threshold.

    Parameters
    ----------
    mc_all : ndarray
        Daily ⟨|C_ij|⟩.
    mean_mc : float
        Global mean of ⟨|C|⟩.

    Returns
    -------
    (list of (int, int), ndarray)
        Windows in chronological order, and the smoothed series.
    """
    intervals_raw, mc_smooth = increasing_intervals(mc_all)
    chrono = sorted([(a, b) for a, b in intervals_raw
                     if np.nanmean(mc_all[a:b + 1]) > config.CRISIS_FACTOR * mean_mc],
                    key=lambda ab: ab[0])
    return chrono, mc_smooth


# ── Loop 2: per-asset crisis (rising E_i) — cached ────────────────────────────
def _crisis_map_from(E):
    """Map (N x days) of the runs where each E_i rises above its own threshold."""
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
    """Per-asset crisis map (|C|), cached under the detection parameters.

    Returns
    -------
    ndarray of bool, shape (N, days)
    """
    sig = (f"af{config.ASSET_FACTOR}_s{config.SMOOTH}_ml{config.MIN_LEN}"
           f"_mg{config.MERGE_GAP}_rf{config.REFINE}")
    return disk_cache("crisis_map", sig, lambda: _crisis_map_from(E_daily_all))


# ── Signed variant (the "without absolute value" experiment) ──────────────────
def compute_E_signed(data, N):
    """SIGNED daily E_i (sum of C_ij without ``np.abs``), cached.

    Parameters
    ----------
    data : DataFrame
        Log-returns (timestamped index).
    N : int
        Number of assets.

    Returns
    -------
    ndarray, shape (N, days)
    """
    def _compute():
        """Compute the signed daily E_i table from the raw returns."""
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
            E[:, k] = C.sum(axis=1) / (N - 1)        # SIGNED: no np.abs
        return E

    return disk_cache("E_daily_signed", f"N{N}T{len(data)}", _compute)


def detect_global_windows_signed(E_signed):
    """Global windows + crisis map on the SIGNED correlation.

    Returns
    -------
    (mc_signed, mc_signed_smooth, intervals_signed, crisis_map_signed)
    """
    mc_signed = np.nanmean(E_signed, axis=0)         # = mean of the signed C_ij
    mean_mc_signed = np.nanmean(mc_signed)
    iv_raw, mc_smooth = increasing_intervals(mc_signed)
    intervals = sorted([(a, b) for a, b in iv_raw
                        if np.nanmean(mc_signed[a:b + 1]) > config.CRISIS_FACTOR * mean_mc_signed],
                       key=lambda ab: ab[0])
    crisis_map_signed = _crisis_map_from(E_signed)
    return mc_signed, mc_smooth, intervals, crisis_map_signed


# ── Detection from the peaks of ⟨|C|⟩ ──────────────────────────────────────────
def detect_peaks(mc_all, mean_mc):
    """Peaks of smoothed ⟨|C|⟩ and each peak's width -> crisis intervals.

    Parameters
    ----------
    mc_all : ndarray
        Daily ⟨|C_ij|⟩.
    mean_mc : float
        Global mean (used as a minimum peak height).

    Returns
    -------
    dict
        ``mc_s`` (smoothed series), ``peaks`` (summit indices),
        ``w_h`` / ``l_ips`` / ``r_ips`` (width height and bounds),
        ``peak_intervals`` (list of (left, right, summit)).
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
    """Peak rises: from the preceding trough to the summit (>= MIN_LEN days).

    Parameters
    ----------
    peak_intervals : list of (int, int, int)
        ``peak_intervals`` output of :func:`detect_peaks`.
    mc_s : ndarray
        Smoothed ⟨|C|⟩ (to locate the troughs).

    Returns
    -------
    list of (int, int)
        Rises (trough -> summit), sorted and deduplicated.
    """
    rises, prev = [], 0
    for a, b, pk in peak_intervals:
        trough = prev + int(np.argmin(mc_s[prev:pk + 1])) if pk > prev else pk
        if pk - trough + 1 >= config.MIN_LEN:
            rises.append((trough, pk))
        prev = pk
    return sorted(set(rises))


# ── Spectral diagnostic λ_max/⟨λ⟩ (market mode) ───────────────────────────────
def spectral_diagnostics(data, N):
    """λ_max/⟨λ⟩ and max|ρ|/⟨|ρ|⟩ in a sliding window, cached.

    Parameters
    ----------
    data : DataFrame
        Log-returns.
    N : int
        Number of assets.

    Returns
    -------
    DataFrame
        Dated index; columns ``ratio_corr, ratio_eig, mean_return,
        max_corr, mean_corr``.
    """
    import pandas as pd

    def _compute():
        """Compute the sliding-window spectral diagnostics table."""
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
    """Dates of the peaks of smoothed λ_max/⟨λ⟩.

    Parameters
    ----------
    diag : DataFrame
        Output of :func:`spectral_diagnostics`.

    Returns
    -------
    DatetimeIndex
        Dates of the λ_max/⟨λ⟩ summits.
    """
    re = uniform_filter1d(diag["ratio_eig"].values, size=config.SPEC_SMOOTH, mode="nearest")
    pk, _ = find_peaks(re, prominence=config.SPEC_PROM_FACTOR * diag["ratio_eig"].std(),
                       distance=20)
    return diag.index[pk]


def mix_periods(diag):
    """"Mix" periods: bumps of the score z(λ_max/⟨λ⟩) + z(max|ρ|/⟨|ρ|⟩).

    Parameters
    ----------
    diag : DataFrame
        Output of :func:`spectral_diagnostics`.

    Returns
    -------
    list of (Timestamp, Timestamp)
        Bounds (left, right) of each bump of the combined score.
    """
    def _z(s):
        """Z-score of a series (NaN-aware)."""
        s = np.asarray(s, float)
        return (s - np.nanmean(s)) / np.nanstd(s)

    mix = uniform_filter1d(_z(diag["ratio_eig"]) + _z(diag["ratio_corr"]),
                           size=config.SPEC_SMOOTH, mode="nearest")
    mpk, _ = find_peaks(mix, prominence=0.8 * np.nanstd(mix), distance=20)
    _lw, _wh, li, ri = peak_widths(mix, mpk, rel_height=0.5)
    return sorted({(diag.index[int(np.floor(l))],
                    diag.index[min(int(np.ceil(r)), len(diag) - 1)])
                   for l, r in zip(li, ri)})


# ═══════════════════════════ SIS dynamics (was sis.py) ═══════════════════════════

"""SIS dynamics — the "one that solves the SIS dynamics given the matrix" block.

The bounded SIS model

    dx_i/dt = -B x_i + R (1 - x_i) Σ_j A_ij x_j ,   x_i ∈ [0, 1]

is integrated from an initial condition x0 (E_i profile on the first day of the
rise), for any contagion matrix ``A`` (PMFG, VAR, Corr thr). The trajectory
x_i(t) is reframed on the convergence time T_conv towards the equilibrium x*,
then compared to the empirical profile E_i(t) by regression -> R².

Three integrators of decreasing fidelity (all equivalent at the fixed point,
only the cost differs):
  - :func:`integrate`        : double integration, 5000-pt grid (reference)
  - :func:`integrate_fast`   : single integration, 5000-pt grid (~2x faster)
  - :func:`integrate_xscan`  : loosened tolerances, 400-pt grid (sweeps)

`fit_periods` computes and caches, for a list of windows, the SIS fits of each
method.
"""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress

import config
from config import disk_cache, sig_of


def _sis_params_tag():
    """Readable suffix of the SIS parameters shared by the cache signatures."""
    return (f"thr{config.CORR_THRESHOLD}_T{config.T_LONG}_B{config.B_FIT}"
            f"_R{config.R_FIT}_tol{config.TOL_EQ}"
            f"_sr{config.SR_SMOOTH}-{config.SR_MIN_LEN}-{config.SR_MIN_GAIN}")


def sig_period_data(N, windows, crisis_map):
    """Signature of the 'period_data' cache (rising windows, |C|)."""
    return f"N{N}_p{len(windows)}_{_sis_params_tag()}_{sig_of(windows, crisis_map)}"


def sig_period_data_signed(windows_signed, crisis_map_signed):
    """Signature of the 'period_data_sig' cache (signed detection, no np.abs)."""
    return (f"SIG_p{len(windows_signed)}_{_sis_params_tag()}"
            f"_{sig_of(windows_signed, crisis_map_signed)}")


def sig_peak_period_data(peak_rises_list, crisis_map):
    """Signature of the 'peak_period_data' cache (peak rises)."""
    return sig_of(peak_rises_list, crisis_map, config.CORR_THRESHOLD, config.T_LONG,
                  config.B_FIT, config.R_FIT, config.TOL_EQ,
                  config.SR_SMOOTH, config.SR_MIN_LEN, config.SR_MIN_GAIN)


def ode_sis_bounded(t, x, A, B, R):
    """Velocity field of the bounded SIS: state clipped to [0, 1].

    The clipping keeps the dynamics stable even for a signed initial condition.

    Parameters
    ----------
    t : float
        Time (ignored, autonomous system).
    x : ndarray
        Current state.
    A : ndarray, shape (N, N)
        Contagion matrix.
    B, R : float
        Recovery and infection rates.

    Returns
    -------
    ndarray
        dx/dt.
    """
    xc = np.clip(x, 0.0, 1.0)
    return -B * xc + R * (1.0 - xc) * (A @ xc)


def _solve(x0, A, t_span, t_eval=None, rtol=1e-6, atol=1e-9, B=None, R=None):
    """``solve_ivp`` shortcut (LSODA); B / R default to the config values."""
    B = config.B_FIT if B is None else B
    R = config.R_FIT if R is None else R
    return solve_ivp(ode_sis_bounded, t_span, x0, args=(A, B, R),
                     t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol)


def integrate(cache_gi, gi, A, B=None, R=None):
    """Trajectory x_i(t) (reference) reframed on the convergence time.

    x* is obtained by a first integration up to ``T_LONG``; the convergence time
    T_conv is the first instant where x_i reaches ``(1 - TOL_EQ) x*``.

    Parameters
    ----------
    cache_gi : dict
        Entry ``cache[gi]``: ``x0`` (initial condition) and ``n_days``.
    gi : int
        Global index of the tracked asset.
    A : ndarray, shape (N, N)
        Contagion matrix.
    B, R : float or None
        Recovery / infection rates; ``None`` -> config values.

    Returns
    -------
    ndarray, shape (n_days,)
        Trajectory x_i reframed and clipped to ]0, 1[.
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
    """Like :func:`integrate` but without the first integration (x* = last point).

    Result identical at the fixed point, ~2x faster; used for the sweeps
    (q-threshold and B/R rates). ``B`` / ``R`` at ``None`` -> config values.
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
    """Lightweight integrator (rtol 1e-4, 400-pt grid) for the VAR-lag sweep."""
    x0, n = cache_gi["x0"], cache_gi["n_days"]
    kw = dict(rtol=1e-4, atol=1e-7, B=B, R=R)
    x_eq = _solve(x0, A, (0, config.T_LONG), **kw).y[gi, -1]
    tp = np.linspace(1e-3, config.T_LONG, 400)
    sp = _solve(x0, A, (0, config.T_LONG), t_eval=tp, **kw)
    above = np.where(sp.y[gi] >= (1 - config.TOL_EQ) * x_eq)[0]
    Tc = float(tp[above[0]]) if len(above) else config.T_LONG
    s = _solve(x0, A, (0, Tc), t_eval=np.linspace(0, Tc, n), **kw)
    return np.clip(s.y[gi], 1e-8, 1 - 1e-8)


# ── Fit context (data + crisis map + matrices) ────────────────────────────────
def make_fit_context(ctx, crisis_map, E_daily=None):
    """Pack what :func:`fits_for` needs (reusable for the signed case).

    Parameters
    ----------
    ctx : SimpleNamespace
        Output of ``data.build_context``.
    crisis_map : ndarray of bool
        Per-asset crisis map (|C| or signed).
    E_daily : ndarray or None
        Daily E_i profiles; ``None`` -> ``ctx.E_daily_all`` (|C| case).

    Returns
    -------
    SimpleNamespace
        Fields ``data, all_days, N, crisis_map, E_daily, A_sis``.
    """
    return SimpleNamespace(data=ctx.data, all_days=ctx.all_days, N=ctx.N,
                           crisis_map=crisis_map,
                           E_daily=ctx.E_daily_all if E_daily is None else E_daily,
                           A_sis=ctx.A_sis)


def cache_signed(pa, pb, fc):
    """For each in-crisis asset on (pa, pb): measured E_i + initial condition x0.

    An asset is kept only if it shows a strong rise (``strong_rise``) in its
    crisis window overlapping (pa, pb).

    Parameters
    ----------
    pa, pb : int
        Bounds (day indices) of the global period.
    fc : SimpleNamespace
        Fit context (:func:`make_fit_context`).

    Returns
    -------
    dict
        ``{gi: dict(E_i, x0, n_days)}`` for the kept assets.
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
    """SIS fits of every method on the period (pa, pb).

    Parameters
    ----------
    pa, pb : int
        Bounds of the global period.
    fc : SimpleNamespace
        Fit context.

    Returns
    -------
    (dict, dict)
        ``cache`` (output of :func:`cache_signed`) and ``fits``:
        ``fits[m] = (pad, sl)`` where ``pad[gi] = (E_i, x_traj)`` and
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
    """Compute (and cache) ``{(pa, pb): fits_for(pa, pb)}`` over windows.

    Parameters
    ----------
    name : str
        Cache prefix ('period_data', 'peak_period_data', 'period_data_sig').
    sig : str
        Cache signature (must reflect windows + crisis map + SIS params).
    windows : list of (int, int)
        Periods to fit.
    fc : SimpleNamespace
        Fit context.

    Returns
    -------
    dict
        ``{(pa, pb): (cache, fits)}``.
    """
    return disk_cache(name, sig,
                      lambda: {(pa, pb): fits_for(pa, pb, fc) for (pa, pb) in windows})
