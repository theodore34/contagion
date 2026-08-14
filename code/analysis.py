"""Reading the SIS fits: R² tables, selection of periods and curves.

From the fit dictionaries produced by :mod:`model` (``{(pa, pb): (cache, fits)}``),
this section:
  - builds the summary tables (1 row per asset x period) -> CSV;
  - selects the periods carrying at least one good fit (R² > threshold);
  - gathers the retained curves for the figures;
  - computes the redundancy recaps and the window summaries.
"""
import numpy as np
import pandas as pd

import config
from model import asset_runs, strong_rise

R2_COLS = [f"R2_{m}" for m in config.SIS_MODELS]


def crisis_table(windows, period_data, ctx, crisis_map, csv_path=None):
    """Table with 1 row per in-crisis (asset, period); R² per method -> CSV.

    Parameters
    ----------
    windows : list of (int, int)
        Global windows (intervals_chrono).
    period_data : dict
        ``{(pa, pb): (cache, fits)}`` from :func:`model.fit_periods`.
    ctx : SimpleNamespace
        Context (asset_names, all_days, E_daily_all).
    crisis_map : ndarray of bool
        Per-asset crisis map (|C|).
    csv_path : str or Path or None
        If provided, write the table to CSV.

    Returns
    -------
    DataFrame
        Sorted by period then decreasing max R².
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
    """Periods carrying at least one fit with R² > ``r2_seuil`` (any method).

    Returns
    -------
    list of (int, int)
    """
    return [(pa, pb) for (pa, pb) in windows
            if any(sl["r2"] > r2_seuil
                   for m in config.SIS_MODELS
                   for sl in period_data[(pa, pb)][1][m][1].values())]


def collect_by_method(period_data, selection, r2_seuil=config.R2_SEUIL):
    """Gather the curves (R² > threshold) per method for the "3 views" figures.

    Parameters
    ----------
    period_data : dict
        Fits per period.
    selection : list of (int, int)
        Retained periods.
    r2_seuil : float
        Fit-selection threshold.

    Returns
    -------
    dict
        ``{method: dict(Es, Xs, E_all, X_all, r2s)}``.
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
    """Peak-rise table (best method per asset) + retained curves.

    Parameters
    ----------
    peak_rises_list : list of (int, int)
        Peak rises (trough -> summit).
    peak_pd : dict
        SIS fits per rise.
    ctx : SimpleNamespace
        Context (asset_names, all_days).
    r2_seuil : float
        Threshold to keep a curve.
    csv_path : str or Path or None
        If provided, write the table to CSV.

    Returns
    -------
    (DataFrame, dict)
        The table sorted by max R² and the retained curves
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
    """Redundancy (fits R²>threshold / distinct assets) per method: without | with peaks.

    Returns
    -------
    DataFrame
        Index = methods; columns ``fits_sans, actifs_sans, redond_sans,
        fits_avec, actifs_avec, redond_avec``.
    """
    def _rows(pdict):
        """Flatten (method, asset) pairs of the fits above threshold."""
        rows = []
        for (pa, pb), (cache, fits) in pdict.items():
            for m in config.SIS_MODELS:
                for gi, sl in fits[m][1].items():
                    if sl["r2"] > r2_seuil:
                        rows.append((m, ctx.asset_names[gi]))
        return pd.DataFrame(rows, columns=["methode", "actif"])

    S, A = _rows(period_data), _rows(peak_pd)

    def _stat(df, m):
        """(#fits, #distinct assets, redundancy) for method m."""
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


# ── Window summaries for the comparison figure (cell 27) ──────────────────────
def _r2max(fits, gi):
    """Max R² over the methods for asset ``gi`` (0 if absent)."""
    return max((fits[m][1][gi]["r2"] for m in config.SIS_MODELS if gi in fits[m][1]),
               default=0.0)


def window_summaries(windows, pdict, all_days, r2_seuil=config.R2_SEUIL):
    """Per window: in-crisis assets + number of fits R²>threshold (both loops)."""
    out = []
    for (pa, pb) in windows:
        cache, fits = pdict[(pa, pb)]
        n_sup = sum(_r2max(fits, gi) > r2_seuil for gi in cache)
        out.append(dict(a=pd.Timestamp(all_days[pa]), b=pd.Timestamp(all_days[pb]),
                        cache=set(cache), n_crise=len(cache), n_sup=int(n_sup)))
    return sorted(out, key=lambda d: d["a"])


def retain(W, retenue_only=True):
    """Filter the retained windows (>= 1 fit R²>threshold) if ``retenue_only``."""
    return [w for w in W if w["n_sup"] > 0] if retenue_only else W


def totals(W):
    """(#distinct in-crisis assets, total #fits R²>threshold) over a list of windows."""
    crise = set().union(*[w["cache"] for w in W]) if W else set()
    return len(crise), sum(w["n_sup"] for w in W)


# ═══════════════════════════ Endo/exo reflexivity (was endo_exo.py) ═══════════════════════════

"""Reproducible endo/exo score — branching ratio (reflexivity) per period.

Replaces the hand-set "exogenous/endogenous" label with a **score computed
identically over every period**, taken from the literature on endogenous vs
exogenous market shocks (Filimonov & Sornette 2012; Hardiman & Bouchaud 2014).

Idea
----
The **branching ratio** ``n`` of a self-exciting Hawkes process quantifies the
share of **ENDOGENOUS** activity (each move triggering others — internal cascade,
reflexivity) relative to the **EXOGENOUS** activity (external shocks arriving
"from outside"). ``n → 1``: near-critical, highly reflexive market;
``n → 0``: Poisson activity driven by external shocks.

*Model-independent* estimator (Hardiman & Bouchaud 2014): for a stationary
Hawkes process, the **dispersion index** (Fano) of the event counts in bins of
equal duration tends, at large windows, towards ::

    F = Var[N] / E[N]  ->  1 / (1 - n)^2      hence   n = 1 - 1/sqrt(F).

It only needs the mean and variance of the counts: no fragile likelihood fit, so
it is stable on short windows.

Recipe (identical for each period)
----------------------------------
- **Event** = "jump" of an asset: ``|r_{i,t}| > k · median(|r_i|)``, the threshold
  being per-asset (scale-invariant) and estimated over the whole sample -> same
  rule everywhere. ``k = 3`` by default.
- **Count** = number of asset jumps, aggregated **per day** (bin = 1 day).
- ``n = 1 - 1/sqrt(Fano)`` over the window's daily counts.

The **ranking** of the periods is invariant to the threshold ``k`` (Spearman rho
≈ 1 for ``k ∈ [2.5, 3.5]``) and to the bin size >= 1 day. At the intra-bar scale
(30 min), ``n ≈ 0.84`` for *all* windows: this is the "apparent criticality"
described by Hardiman & Bouchaud — it is at the **daily** step that the branching
ratio distinguishes the crises.

References
----------
V. Filimonov, D. Sornette, *Quantifying reflexivity in financial markets*,
Phys. Rev. E **85**, 056108 (2012).
S. J. Hardiman, J.-P. Bouchaud, *Branching-ratio approximation for the
self-exciting Hawkes process*, Phys. Rev. E **90**, 062807 (2014).
"""
import numpy as np
import pandas as pd

JUMP_K = 3.0        # an asset "jumps" if |r| > JUMP_K x its median of |r|


def market_jump_activity(data, k=JUMP_K):
    """Market jump activity, bar by bar.

    Parameters
    ----------
    data : DataFrame, shape (T, N)
        Intraday log-returns.
    k : float
        An asset jumps at bar ``t`` if ``|r_{i,t}| > k · median(|r_i|)``
        (per-asset scale, estimated over the whole sample).

    Returns
    -------
    ndarray, shape (T,)
        Number of jumping assets at each bar.
    """
    R = np.abs(np.asarray(data.values, float))
    med = np.nanmedian(R, axis=0)
    med[med == 0] = np.nanmedian(med[med > 0])
    return (R > k * med).sum(axis=1).astype(float)


def _day_of_bar(data, all_days):
    """Day index (in ``all_days``) of each bar of ``data``."""
    bar_day = pd.to_datetime(data.index).normalize().values
    return np.searchsorted(np.asarray(all_days), bar_day, side="right") - 1


def branching_ratio(counts):
    """Branching ratio ``n = 1 - 1/sqrt(Var/Mean)`` over per-bin counts.

    *Model-independent* estimator of Hardiman & Bouchaud (2014): for a stationary
    Hawkes process, the dispersion index of the counts tends to ``1/(1-n)^2``.

    Parameters
    ----------
    counts : array-like
        Number of events per time bin (bins of equal duration).

    Returns
    -------
    float
        ``n`` in ``[0, 1)``; ``nan`` if fewer than 5 bins or zero mean.
        Under-dispersion (``F <= 1``, ~Poisson case) -> ``0``.
    """
    c = np.asarray(counts, float)
    c = c[np.isfinite(c)]
    if len(c) < 5 or c.mean() <= 0:
        return np.nan
    F = c.var(ddof=1) / c.mean()
    return float(max(0.0, 1.0 - 1.0 / np.sqrt(F))) if F > 1 else 0.0


def _daily_counts(activity, day_of_bar, a, b):
    """Jump counts aggregated per day over the day window ``[a, b]``."""
    m = (day_of_bar >= a) & (day_of_bar <= b)
    if not m.any():
        return np.array([])
    return pd.Series(activity[m]).groupby(day_of_bar[m]).sum().values


def reflexivity_by_period(ctx, periods_list, k=JUMP_K, n_boot=2000, seed=0):
    """Branching ratio ``n`` (reflexivity) per period, with bootstrap CI.

    Computed identically over each window: events = asset jumps
    (``|r| > k · median``), bin = 1 day, ``n = 1 - 1/sqrt(Fano)``.

    Parameters
    ----------
    ctx : SimpleNamespace
        Shared context (``data``, ``all_days``).
    periods_list : list of (int, int)
        Windows ``(start, end)`` in day indices (typically ``peak_rises``).
    k : float
        Jump threshold (x per-asset median of ``|r|``).
    n_boot : int
        Number of bootstrap resamples of the daily counts (95% CI).
    seed : int
        Seed of the pseudo-random generator.

    Returns
    -------
    DataFrame
        Columns ``periode, n, n_lo, n_hi, n_sd, Fano, n_days, act_per_day``
        (``n_sd`` = bootstrap std of ``n``; one row per period, chronological
        input order).
    """
    all_days = np.asarray(pd.to_datetime(ctx.all_days))
    activity = market_jump_activity(ctx.data, k)
    dob = _day_of_bar(ctx.data, all_days)
    rng = np.random.default_rng(seed)
    rows = []
    for (a, b) in periods_list:
        c = _daily_counts(activity, dob, a, b)
        n = branching_ratio(c)
        if len(c) >= 5 and np.isfinite(n):
            boots = [branching_ratio(rng.choice(c, len(c), replace=True))
                     for _ in range(n_boot)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            sd = float(np.nanstd(boots, ddof=1))
            F = float(c.var(ddof=1) / c.mean())
        else:
            lo = hi = sd = F = np.nan
        rows.append(dict(
            periode=f"{str(pd.Timestamp(all_days[a]))[:10]} → {str(pd.Timestamp(all_days[b]))[:10]}",
            n=n, n_lo=lo, n_hi=hi, n_sd=sd, Fano=F, n_days=int(len(c)),
            act_per_day=float(np.mean(c)) if len(c) else np.nan))
    return pd.DataFrame(rows)


def reflexivity_series(ctx, win_days=21, k=JUMP_K):
    """Sliding series of the branching ratio ``n`` (bin = day).

    ``n`` estimated on the last ``win_days`` days at each date (≈ 1 month,
    consistent with ``WINDOW_SPEC``). Used to plot reflexivity over time, in the
    style of the spectral diagnostic λ_max/⟨λ⟩.

    Parameters
    ----------
    ctx : SimpleNamespace
        Shared context (``data``, ``all_days``).
    win_days : int
        Width of the sliding window, in days.
    k : float
        Jump threshold (x per-asset median of ``|r|``).

    Returns
    -------
    Series
        ``n`` indexed by date (``nan`` before the first complete window).
    """
    all_days = np.asarray(pd.to_datetime(ctx.all_days))
    activity = market_jump_activity(ctx.data, k)
    dob = _day_of_bar(ctx.data, all_days)
    daily = (pd.Series(activity).groupby(dob).sum()
             .reindex(range(len(all_days)), fill_value=0).values)
    n = np.full(len(all_days), np.nan)
    for d in range(win_days, len(all_days) + 1):
        n[d - 1] = branching_ratio(daily[d - win_days:d])
    return pd.Series(n, index=pd.to_datetime(all_days), name="n_reflex")


def endo_exo_index(n_values, k=5):
    """Integer endo/exo index from 1 to ``k`` by quantiles of the score ``n``.

    Since the score ``n`` is high and tight everywhere (near-critical market), the
    index is **comparative**: it splits the periods into ``k`` groups of ~equal
    sizes by score rank. ``1`` = most **exogenous** (lowest ``n``), ``k`` = most
    **endogenous** (highest ``n``).

    Parameters
    ----------
    n_values : array-like
        The ``n`` values of the periods.
    k : int
        Number of levels of the scale (5 by default).

    Returns
    -------
    list of int
        The index ``1..k`` of each period (``0`` if ``n`` is ``nan``).
    """
    n = pd.Series(n_values, dtype=float)
    ranks = n.rank(method="average")
    idx = np.ceil(ranks / n.notna().sum() * k).clip(1, k)
    return [int(v) if np.isfinite(v) else 0 for v in idx]


def tag_from_terciles(n_values):
    """Comparative endo/exo label by terciles of the score ``n`` (17 periods).

    Since the score ``n`` is high everywhere (near-critical market), the label is
    **comparative**: it situates each period relative to the others.

    Parameters
    ----------
    n_values : array-like
        The ``n`` values of the periods.

    Returns
    -------
    list of str
        ``'endogène (réflexivité forte)'`` / ``'intermédiaire'`` /
        ``'exogène (réflexivité faible)'`` depending on the tercile.
    """
    n = np.asarray(n_values, float)
    q1, q2 = np.nanquantile(n, [1 / 3, 2 / 3])
    out = []
    for v in n:
        if not np.isfinite(v):
            out.append("?")
        elif v >= q2:
            out.append("endogène (réflexivité forte)")
        elif v <= q1:
            out.append("exogène (réflexivité faible)")
        else:
            out.append("intermédiaire")
    return out


# ═══════════════════════════ Null model (was null_random_data.py) ═══════════════════════════

"""Null model 'random data, same matrices': block-bootstrap of the returns,
re-detection + SIS fit with the REAL A matrices, count of the R2>threshold curves.

The block-bootstrap keeps the contemporaneous correlation (so ~as many detected
curves) but breaks the temporal order -> tests whether the real dynamics matter.
iid / phase-rand destroy the correlation (0 curve), confirmed separately.

Output: cache/null_random_blockboot.pkl (real_counts, null_counts per method).
"""
import sys, pickle, numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.stats import linregress
from joblib import Parallel, delayed

import config
import data as datamod
import model

# Context loaded on demand (NOT at import of analysis):
N = MASK = SIS_MODELS = A_SIS = REALVALS = IDX = COLS = None
R2 = config.R2_SEUIL


def _ensure_null():
    """Load the null-model context on first call (module globals)."""
    global N, MASK, SIS_MODELS, A_SIS, REALVALS, IDX, COLS
    if N is not None:
        return
    ctx = datamod.build_context()
    N, MASK = ctx.N, ctx.mask_off
    SIS_MODELS = list(ctx.A_sis.keys())
    A_SIS = ctx.A_sis
    REALVALS = ctx.data.values
    IDX, COLS = ctx.data.index, ctx.data.columns


def _light_base(vals):
    """days, daily mc, daily E_i from a (T, N) array aligned on IDX."""
    df = pd.DataFrame(vals, index=IDX, columns=COLS)
    day = np.array([t.date() for t in IDX])
    days = np.array(sorted(set(day)))
    mc = np.full(len(days), np.nan); E = np.full((N, len(days)), np.nan)
    for k, dd in enumerate(days):
        chunk = vals[day == dd]
        if len(chunk) < 3:
            continue
        C = np.abs(np.nan_to_num(np.corrcoef(chunk.T), nan=0.0)); np.fill_diagonal(C, 0)
        mc[k] = C[MASK].mean(); E[:, k] = C.sum(1) / (N - 1)
    return df, days, mc, E


def _count(vals):
    """Number of R2>threshold curves per method (real A matrices, integrate_xscan)."""
    _ensure_null()
    df, days, mc, E = _light_base(vals)
    cm = model._crisis_map_from(E)
    pk = model.detect_peaks(mc, float(np.nanmean(mc)))
    pr = model.peak_rises(pk['peak_intervals'], pk['mc_s'])
    fc = SimpleNamespace(data=df, all_days=days, N=N, crisis_map=cm, E_daily=E, A_sis=A_SIS)
    cnt = {m: 0 for m in SIS_MODELS}; nc = 0
    for (pa, pb) in pr:
        cache = model.cache_signed(pa, pb, fc)
        for gi in cache:
            E_i = cache[gi]['E_i']; ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            nc += 1
            for m in SIS_MODELS:
                xt = model.integrate_xscan(cache[gi], gi, A_SIS[m])
                if linregress(E_i[ok], xt[ok]).rvalue ** 2 > R2:
                    cnt[m] += 1
    return dict(n_curves=nc, n_peaks=len(pr), counts=cnt)


def _block_boot(seed, block=20):
    """Block-bootstrap of the real returns (keeps the contemporaneous correlation)."""
    _ensure_null()
    rng = np.random.default_rng(seed)
    T = REALVALS.shape[0]; nb = T // block
    starts = rng.integers(0, T - block + 1, size=nb)
    out = np.vstack([REALVALS[s:s + block] for s in starts])[:T]
    if out.shape[0] < T:                      # complete to T rows
        out = np.vstack([out, REALVALS[:T - out.shape[0]]])
    return out


def _one_null(seed):
    """One null realisation: block-bootstrap then count."""
    return _count(_block_boot(seed))


if __name__ == '__main__':
    _ensure_null()
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    real = _count(REALVALS)
    print('REAL :', real, flush=True)
    res = Parallel(n_jobs=min(NB, config.N_JOBS))(delayed(_one_null)(s) for s in range(NB))
    null = {m: np.array([r['counts'][m] for r in res], float) for m in SIS_MODELS}
    print(f'\n{NB} block-bootstraps (courbes ~{np.mean([r["n_curves"] for r in res]):.0f}/realisation)')
    print(f"{'methode':10s} {'reel':>5s} {'null_moy':>8s} {'null_sd':>7s} {'z':>6s} {'p':>8s}")
    summary = {'real': real['counts'], 'null': {}}
    for m in SIS_MODELS:
        mu, sd = null[m].mean(), null[m].std()
        z = (real['counts'][m] - mu) / (sd + 1e-12)
        p = (1 + np.sum(null[m] >= real['counts'][m])) / (NB + 1)
        summary['null'][m] = null[m]
        print(f'{m:10s} {real["counts"][m]:5d} {mu:8.1f} {sd:7.2f} {z:6.1f} {p:8.4f}')
    pickle.dump(summary, open('../cache/null_random_blockboot.pkl', 'wb'))
    print('\n-> cache/null_random_blockboot.pkl')


# ═══════════════════════════ ξ_emp calibration (was proxy.py) ═══════════════════════════

"""Calibration of the correlation energy E_i into a stress probability ξ_emp.

Utility functions consumed by the notebooks (see notebooks/article.ipynb).

The SIS produces x_i(t) ∈ [0, 1] = probability that asset i is "infected". To
compare it against an observable of the **same nature**, we calibrate E_i
(correlation energy, ∈ [0, 1] but not a probability) into a probability of
extreme return, via an independent indicator I_i(t) = 1(|r_i(t)| > q_i^90):

    ξ_emp(E-bin) = P(I=1 | E ∈ bin) = #{I=1 in the bin} / #{obs in the bin}.

`build_mapping` returns the function f : E ↦ P(stress | E) (monotone/smooth), and
`map_energy(E, f)` maps any profile E_i(t) into ξ_emp(t) = f(E_i(t)).
"""
from types import SimpleNamespace

import numpy as np


def daily_log_returns(data, all_days):
    """Daily log-returns (N, n_days) aligned on ``all_days``.

    Since the intraday 30-min log-returns are additive, their sum over a day
    gives that day's close-to-close log-return.
    """
    df = data.copy()
    df["day"] = df.index.date
    return df.groupby("day").sum().reindex(all_days).values.T


def extreme_indicator(daily_ret, q=0.90):
    """I_i(t) = 1(|r_i(t)| > q_i^q), threshold q **per asset** (NaN where r missing)."""
    a = np.abs(daily_ret)
    thr = np.nanquantile(a, q, axis=1, keepdims=True)
    I = (a > thr).astype(float)
    I[~np.isfinite(daily_ret)] = np.nan
    return I


def pairs(E, I):
    """Flatten the valid pairs (E_i(t), I_i(t)) → two 1D vectors."""
    ok = np.isfinite(E) & np.isfinite(I)
    return E[ok], I[ok].astype(float)


def calibration_curve(Ev, Iv, bin_width=0.05, min_count=50):
    """ξ_emp(E-bin) = P(I=1 | E ∈ bin) by fixed-width bins.

    Returns a dict ``centers, p, n, se`` (se = binomial std), the bins with count
    < ``min_count`` being dropped.
    """
    edges = np.arange(0.0, Ev.max() + bin_width, bin_width)
    idx = np.digitize(Ev, edges) - 1
    c, p, n, se = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < min_count:
            continue
        pb = float(Iv[m].mean())
        c.append(0.5 * (edges[b] + edges[b + 1])); p.append(pb)
        n.append(int(m.sum())); se.append(np.sqrt(pb * (1 - pb) / m.sum()))
    return dict(centers=np.array(c), p=np.array(p), n=np.array(n), se=np.array(se))


def crossing(x, y, level):
    """First x where y crosses ``level`` (linear interpolation), else None."""
    d = np.asarray(y) - level
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(s) == 0:
        return None
    i = s[0]
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))


def build_mapping(ctx, q=0.90, kind="monotone"):
    """Indicator + pairs + curve + mapping functions E ↦ P(stress|E).

    ``kind`` : ``"monotone"`` (hugs the data, isotonic regression) or
    ``"smooth"`` (smooth, logistic regression).

    Returns a ``SimpleNamespace``: ``f`` (E→prob, according to ``kind``),
    ``monotone`` / ``smooth`` (the two mappings), ``curve``, ``Ev`` / ``Iv``
    (pairs), ``marginal`` (global P(I=1)), ``q``.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    E = ctx.E_daily_all
    R = daily_log_returns(ctx.data, ctx.all_days)
    I = extreme_indicator(R, q)
    Ev, Iv = pairs(E, I)

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(Ev, Iv)
    lr = LogisticRegression().fit(Ev.reshape(-1, 1), Iv.astype(int))
    monotone = lambda e: ir.predict(np.ravel(e))
    smooth = lambda e: lr.predict_proba(np.ravel(e).reshape(-1, 1))[:, 1]

    return SimpleNamespace(f=monotone if kind == "monotone" else smooth,
                           monotone=monotone, smooth=smooth,
                           curve=calibration_curve(Ev, Iv), Ev=Ev, Iv=Iv,
                           marginal=float(Iv.mean()), q=q)


def map_energy(E, f):
    """Apply the calibration f to an array E (shape and NaN preserved)."""
    E = np.asarray(E, dtype=float)
    out = np.full(E.shape, np.nan)
    m = np.isfinite(E)
    if m.any():
        out[m] = f(E[m])
    return out


# ═══════════════════════════ Orchestration (was pipeline.py) ═══════════════════════════

"""Orchestration: chains data → periods → SIS → selection.

A thin, plot-free layer shared by all the ``run_*.py`` scripts. Each function
returns objects already cached by the underlying modules, so calling them several
times (from different scripts) recomputes nothing.
"""
import numpy as np
from scipy.stats import linregress

import config
import model
from config import disk_cache, sig_of
from networks import build_filtered_corr, fit_var, prepare_for_sis


def load_periods(ctx):
    """Loop 1 (global windows) + loop 2 (per-asset crisis map, |C|).

    Returns
    -------
    dict
        ``intervals_chrono, mc_smooth, crisis_map``.
    """
    intervals_chrono, mc_smooth = model.detect_global_windows(ctx.mc_all, ctx.mean_mc)
    crisis_map = model.compute_crisis_map(ctx.E_daily_all)
    return dict(intervals_chrono=intervals_chrono, mc_smooth=mc_smooth, crisis_map=crisis_map)


def fit_main(ctx, intervals_chrono, crisis_map):
    """SIS fits on the rising windows (|C|) + selected periods.

    Returns
    -------
    dict
        ``period_data`` (cache) and ``selected`` (periods with >=1 fit R²>threshold).
    """
    fc = model.make_fit_context(ctx, crisis_map)
    sig = model.sig_period_data(ctx.N, intervals_chrono, crisis_map)
    period_data = model.fit_periods("period_data", sig, intervals_chrono, fc)
    selected = select_periods(intervals_chrono, period_data)
    return dict(period_data=period_data, selected=selected)


def fit_peaks(ctx, crisis_map):
    """Peak detection + rises + SIS fits on the rises.

    Returns
    -------
    dict
        ``peak`` (output of detect_peaks), ``peak_rises``, ``peak_pd`` (cache).
    """
    peak = model.detect_peaks(ctx.mc_all, ctx.mean_mc)
    pr = model.peak_rises(peak["peak_intervals"], peak["mc_s"])
    fc = model.make_fit_context(ctx, crisis_map)
    sig = model.sig_peak_period_data(pr, crisis_map)
    peak_pd = model.fit_periods("peak_period_data", sig, pr, fc)
    return dict(peak=peak, peak_rises=pr, peak_pd=peak_pd)


def fit_signed(ctx):
    """"Without absolute value" pipeline: signed detection + SIS fits.

    Returns
    -------
    dict
        ``E_signed, mc_signed, mc_signed_smooth, intervals_signed,
        crisis_map_signed, period_data_sig, selected_sig``.
    """
    E_signed = model.compute_E_signed(ctx.data, ctx.N)
    mc_signed, mc_signed_smooth, intervals_signed, cm_signed = \
        model.detect_global_windows_signed(E_signed)
    fc = model.make_fit_context(ctx, cm_signed, E_daily=E_signed)
    sig = model.sig_period_data_signed(intervals_signed, cm_signed)
    period_data_sig = model.fit_periods("period_data_sig", sig, intervals_signed, fc)
    selected_sig = select_periods(intervals_signed, period_data_sig)
    return dict(E_signed=E_signed, mc_signed=mc_signed, mc_signed_smooth=mc_signed_smooth,
                intervals_signed=intervals_signed, crisis_map_signed=cm_signed,
                period_data_sig=period_data_sig, selected_sig=selected_sig)


def spectral(ctx):
    """Spectral diagnostic λ_max/⟨λ⟩ + peaks + 'mix' periods (cached)."""
    diag = model.spectral_diagnostics(ctx.data, ctx.N)
    return dict(diag=diag, lam_dates=model.lambda_peaks(diag),
                mix_periods=model.mix_periods(diag))


# ── Experiments ───────────────────────────────────────────────────────────────
def _corr_thr_fits(q, ctx, intervals_chrono, period_data):
    """'Corr thr' fits at quantile q over all in-crisis assets (all periods).

    Reuses the already-cached initial conditions (``period_data[...][0]``),
    independent of q; only the 'Corr thr' matrix depends on q.
    """
    A_q = build_filtered_corr(ctx.C_full, q, ctx.mask_off)
    Es, Xs, E_all, X_all, r2s, keys = [], [], [], [], [], []
    for (pa, pb) in intervals_chrono:
        cache = period_data[(pa, pb)][0]
        for gi in cache:
            E_i = cache[gi]["E_i"]
            xt = model.integrate_fast(cache[gi], gi, A_q)
            ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            f = linregress(E_i[ok], xt[ok])
            Es.append(E_i); Xs.append(xt)
            E_all.append(E_i[ok]); X_all.append(xt[ok])
            r2s.append(f.rvalue ** 2); keys.append((pa, pb, gi))
    return Es, Xs, E_all, X_all, r2s, keys


def run_qsweep(ctx, intervals_chrono, crisis_map, period_data):
    """Sweep of the 'Corr thr' q-threshold (parallelised, cached).

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
        """Run the q-sweep in parallel and index by q."""
        res = Parallel(n_jobs=min(len(config.Q_SWEEP), config.N_JOBS))(
            delayed(_corr_thr_fits)(q, ctx, intervals_chrono, period_data)
            for q in config.Q_SWEEP)
        return {round(float(q), 2): r for q, r in zip(config.Q_SWEEP, res)}

    return disk_cache("qsweep_corrthr", sig, _compute)


def _br_counts(B, R, caches, A_sis, r2_seuil):
    """For a pair (B, R): per method, number of fits and distinct assets R²>threshold.

    Reuses the already-cached initial conditions (``caches``), which do not depend
    on (B, R); only the SIS integration does.

    Parameters
    ----------
    B, R : float
        Recovery and infection rates.
    caches : list of dict
        One ``cache`` entry (gi -> dict(E_i, x0, n_days)) per window.
    A_sis : dict
        Contagion matrices per method.
    r2_seuil : float
        Fit-selection threshold.

    Returns
    -------
    dict
        ``{method: dict(n_fits, n_assets, med_r2_all, mean_r2_all, med_r2_sel,
        agg_r2_all, agg_r2_sel)}``. ``med_*``/``mean_*`` are median/mean per-curve
        R²; ``agg_*`` is the R² of the aggregate regression on the pooled E↔x cloud
        (``_sel`` = fits R²>threshold, ``_all`` = all; ``nan`` if empty).
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
                xt = model.integrate_xscan(d, gi, A, B=B, R=R)   # lightweight integrator (sweep)
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
        """Aggregate R² of the pooled (E, x) cloud (nan if empty)."""
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
    """Sweep the (B, R) grid -> retained assets/fits per method (parallelised, cached).

    Parameters
    ----------
    ctx : SimpleNamespace
        Context (for ``A_sis``).
    windows : list of (int, int)
        Windows to count over (typically ``intervals_chrono``).
    crisis_map : ndarray of bool
        Crisis map (only enters the cache signature).
    period_data : dict
        Already-computed fits (only the initial conditions are reused).
    B_grid, R_grid : array-like
        Swept values of B and R.

    Returns
    -------
    dict
        ``{(B, R): {method: dict(n_fits, n_assets, med_r2_all, mean_r2_all, med_r2_sel)}}``.
    """
    import os
    from joblib import Parallel, delayed

    caches = [period_data[(pa, pb)][0] for (pa, pb) in windows]
    pairs = [(round(float(B), 2), round(float(R), 2)) for B in B_grid for R in R_grid]
    sig = sig_of(np.asarray(B_grid, float), np.asarray(R_grid, float), windows, crisis_map,
                 config.CORR_THRESHOLD, config.T_LONG, config.TOL_EQ, config.R2_SEUIL, "br_fast_r2")

    def _compute():
        """Run the (B, R) scan in parallel and index by pair."""
        res = Parallel(n_jobs=min(len(pairs), config.N_JOBS))(
            delayed(_br_counts)(B, R, caches, ctx.A_sis, config.R2_SEUIL) for (B, R) in pairs)
        return {p: r for p, r in zip(pairs, res)}

    return disk_cache("br_scan", sig, _compute)


def _r2_for_A(A_model, ctx, selected, period_data):
    """Aggregate R² (x_i SIS vs E_i) over 'selected' for a contagion matrix."""
    r2s, Ea, Xa = [], [], []
    for (pa, pb) in selected:
        cache = period_data[(pa, pb)][0]
        for gi in cache:
            E_i = cache[gi]["E_i"]
            ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            xt = model.integrate_xscan(cache[gi], gi, A_model)
            r2s.append(linregress(E_i[ok], xt[ok]).rvalue ** 2)
            Ea.append(E_i[ok]); Xa.append(xt[ok])
    r2s = np.array(r2s)
    agg = linregress(np.concatenate(Ea), np.concatenate(Xa)).rvalue ** 2 if Ea else np.nan
    return dict(n=len(r2s), n_sup=int((r2s > config.R2_SEUIL).sum()),
                R2_moyen=float(np.nanmean(r2s)), R2_agrege=float(agg))


def run_lag_scan(ctx, selected, period_data):
    """VAR-lag sweep -> SIS fit quality (cached).

    Returns
    -------
    DataFrame
        Index = VAR lag; columns ``n, n_sup, R2_moyen, R2_agrege``.
    """
    import pandas as pd

    lags = config.LAG_SCAN

    def _scan():
        """SIS fit quality for each VAR lag."""
        return {d: _r2_for_A(prepare_for_sis(np.abs(fit_var(ctx.data.values, d).T)),
                             ctx, selected, period_data) for d in lags}

    sig = (f"lags{lags[0]}-{lags[-1]}_xscan_sel{len(selected)}_T{config.T_LONG}"
           f"_B{config.B_FIT}_R{config.R_FIT}_tol{config.TOL_EQ}_{sig_of(selected)}")
    lag_res = disk_cache("var_lag_scan", sig, _scan)
    ldf = pd.DataFrame(lag_res).T
    ldf.index.name = "lag_VAR"
    return ldf
