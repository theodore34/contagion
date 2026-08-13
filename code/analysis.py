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
from model import asset_runs, strong_rise

R2_COLS = [f"R2_{m}" for m in config.SIS_MODELS]


def crisis_table(windows, period_data, ctx, crisis_map, csv_path=None):
    """Tableau 1 ligne par (actif, période) en crise ; R² par méthode -> CSV.

    Parameters
    ----------
    windows : list of (int, int)
        Fenêtres globales (intervals_chrono).
    period_data : dict
        ``{(pa, pb): (cache, fits)}`` issu de :func:`model.fit_periods`.
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


# ═══════════════════════════ Réflexivité endo/exo (ex-endo_exo.py) ═══════════════════════════

"""Score endo/exo reproductible — ratio de branchement (réflexivité) par période.

Remplace l'étiquette « exogène/endogène » posée à la main par un **score calculé
identiquement sur toutes les périodes**, tiré de la littérature sur les chocs
endogènes vs exogènes des marchés (Filimonov & Sornette 2012 ; Hardiman &
Bouchaud 2014).

Idée
----
Le **ratio de branchement** ``n`` d'un processus auto-excitant de Hawkes quantifie
la part d'activité **ENDOGENE** (chaque mouvement en déclenche d'autres — cascade
interne, réflexivité) par rapport à l'activité **EXOGENE** (chocs externes qui
arrivent « de l'extérieur »). ``n → 1`` : marché quasi-critique, très réflexif ;
``n → 0`` : activité poissonienne dirigée par des chocs externes.

Estimateur *model-independent* (Hardiman & Bouchaud 2014) : pour un Hawkes
stationnaire, l'**indice de dispersion** (Fano) des comptages d'événements dans
des bins de même durée tend, à grande fenêtre, vers ::

    F = Var[N] / E[N]  ->  1 / (1 - n)^2      d'où   n = 1 - 1/sqrt(F).

Il ne demande que la moyenne et la variance des comptages : pas d'ajustement de
vraisemblance fragile, donc stable sur des fenêtres courtes.

Recette (identique pour chaque période)
---------------------------------------
- **Événement** = « saut » d'un actif : ``|r_{i,t}| > k · médiane(|r_i|)``, le
  seuil étant propre à chaque actif (invariant d'échelle) et estimé sur tout
  l'échantillon -> même règle partout. ``k = 3`` par défaut.
- **Comptage** = nombre de sauts d'actifs, agrégé **par jour** (bin = 1 jour).
- ``n = 1 - 1/sqrt(Fano)`` sur les comptages journaliers de la fenêtre.

Le **classement** des périodes est invariant au seuil ``k`` (rho de Spearman ≈ 1
pour ``k ∈ [2.5, 3.5]``) et à la taille de bin ≥ 1 jour. À l'échelle intra-barre
(30 min), ``n ≈ 0.84`` pour *toutes* les fenêtres : c'est la « criticité
apparente » décrite par Hardiman & Bouchaud — c'est au pas **journalier** que le
branching ratio différencie les crises.

Références
----------
V. Filimonov, D. Sornette, *Quantifying reflexivity in financial markets*,
Phys. Rev. E **85**, 056108 (2012).
S. J. Hardiman, J.-P. Bouchaud, *Branching-ratio approximation for the
self-exciting Hawkes process*, Phys. Rev. E **90**, 062807 (2014).
"""
import numpy as np
import pandas as pd

JUMP_K = 3.0        # un actif « saute » si |r| > JUMP_K x sa médiane de |r|


def market_jump_activity(data, k=JUMP_K):
    """Activité de saut du marché, barre par barre.

    Parameters
    ----------
    data : DataFrame, shape (T, N)
        Log-rendements intraday.
    k : float
        Un actif saute à la barre ``t`` si ``|r_{i,t}| > k · médiane(|r_i|)``
        (échelle propre à chaque actif, estimée sur tout l'échantillon).

    Returns
    -------
    ndarray, shape (T,)
        Nombre d'actifs qui sautent à chaque barre.
    """
    R = np.abs(np.asarray(data.values, float))
    med = np.nanmedian(R, axis=0)
    med[med == 0] = np.nanmedian(med[med > 0])
    return (R > k * med).sum(axis=1).astype(float)


def _day_of_bar(data, all_days):
    """Index de jour (dans ``all_days``) de chaque barre de ``data``."""
    bar_day = pd.to_datetime(data.index).normalize().values
    return np.searchsorted(np.asarray(all_days), bar_day, side="right") - 1


def branching_ratio(counts):
    """Ratio de branchement ``n = 1 - 1/sqrt(Var/Mean)`` sur des comptages par bin.

    Estimateur *model-independent* de Hardiman & Bouchaud (2014) : pour un
    processus de Hawkes stationnaire, l'indice de dispersion des comptages tend
    vers ``1/(1-n)^2``.

    Parameters
    ----------
    counts : array-like
        Nombre d'événements par bin de temps (bins de même durée).

    Returns
    -------
    float
        ``n`` dans ``[0, 1)`` ; ``nan`` si moins de 5 bins ou moyenne nulle.
        Sous-dispersion (``F <= 1``, cas ~poissonien) -> ``0``.
    """
    c = np.asarray(counts, float)
    c = c[np.isfinite(c)]
    if len(c) < 5 or c.mean() <= 0:
        return np.nan
    F = c.var(ddof=1) / c.mean()
    return float(max(0.0, 1.0 - 1.0 / np.sqrt(F))) if F > 1 else 0.0


def _daily_counts(activity, day_of_bar, a, b):
    """Comptages de sauts agrégés par jour sur la fenêtre de jours ``[a, b]``."""
    m = (day_of_bar >= a) & (day_of_bar <= b)
    if not m.any():
        return np.array([])
    return pd.Series(activity[m]).groupby(day_of_bar[m]).sum().values


def reflexivity_by_period(ctx, periods_list, k=JUMP_K, n_boot=2000, seed=0):
    """Ratio de branchement ``n`` (réflexivité) par période, avec IC bootstrap.

    Calculé identiquement sur chaque fenêtre : événements = sauts d'actifs
    (``|r| > k · médiane``), bin = 1 jour, ``n = 1 - 1/sqrt(Fano)``.

    Parameters
    ----------
    ctx : SimpleNamespace
        Contexte partagé (``data``, ``all_days``).
    periods_list : list of (int, int)
        Fenêtres ``(début, fin)`` en index de jour (typiquement ``peak_rises``).
    k : float
        Seuil de saut (x médiane de ``|r|`` par actif).
    n_boot : int
        Nombre de rééchantillonnages bootstrap des comptages journaliers (IC 95 %).
    seed : int
        Graine du générateur pseudo-aléatoire.

    Returns
    -------
    DataFrame
        Colonnes ``periode, n, n_lo, n_hi, n_sd, Fano, n_days, act_per_day``
        (``n_sd`` = écart-type bootstrap de ``n`` ; une ligne par période, ordre
        chronologique d'entrée).
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
    """Série glissante du ratio de branchement ``n`` (bin = jour).

    ``n`` estimé sur les ``win_days`` derniers jours à chaque date (≈ 1 mois,
    cohérent avec ``WINDOW_SPEC``). Sert à tracer la réflexivité dans le temps,
    à la façon du diagnostic spectral λ_max/⟨λ⟩.

    Parameters
    ----------
    ctx : SimpleNamespace
        Contexte partagé (``data``, ``all_days``).
    win_days : int
        Largeur de la fenêtre glissante, en jours.
    k : float
        Seuil de saut (x médiane de ``|r|`` par actif).

    Returns
    -------
    Series
        ``n`` indexé par date (``nan`` avant la première fenêtre complète).
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
    """Indice endo/exo entier de 1 à ``k`` par quantiles du score ``n``.

    Le score ``n`` étant élevé et resserré partout (marché quasi-critique),
    l'indice est **comparatif** : il répartit les périodes en ``k`` groupes de
    tailles ~égales par rang du score. ``1`` = le plus **exogène** (``n`` le plus
    faible), ``k`` = le plus **endogène** (``n`` le plus fort).

    Parameters
    ----------
    n_values : array-like
        Les valeurs de ``n`` des périodes.
    k : int
        Nombre de niveaux de l'échelle (5 par défaut).

    Returns
    -------
    list of int
        L'indice ``1..k`` de chaque période (``0`` si ``n`` est ``nan``).
    """
    n = pd.Series(n_values, dtype=float)
    ranks = n.rank(method="average")
    idx = np.ceil(ranks / n.notna().sum() * k).clip(1, k)
    return [int(v) if np.isfinite(v) else 0 for v in idx]


def tag_from_terciles(n_values):
    """Étiquette comparative endo/exo par terciles du score ``n`` (17 périodes).

    Le score ``n`` étant élevé partout (marché quasi-critique), l'étiquette est
    **comparative** : elle situe chaque période par rapport aux autres.

    Parameters
    ----------
    n_values : array-like
        Les valeurs de ``n`` des périodes.

    Returns
    -------
    list of str
        ``'endogène (réflexivité forte)'`` / ``'intermédiaire'`` /
        ``'exogène (réflexivité faible)'`` selon le tercile.
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


# ═══════════════════════════ Modèle nul (ex-null_random_data.py) ═══════════════════════════

"""Null model 'donnees random, memes matrices' : block-bootstrap des rendements,
re-detection + fit SIS avec les VRAIES matrices A, comptage des courbes R2>seuil.

Le block-bootstrap conserve la correlation contemporaine (donc ~autant de courbes
detectees) mais casse l'ordre temporel -> teste si la dynamique reelle compte.
iid / phase-rand detruisent la correlation (0 courbe), confirmes a part.

Sortie : cache/null_random_blockboot.pkl (real_counts, null_counts par methode).
"""
import sys, pickle, numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.stats import linregress
from joblib import Parallel, delayed

import config
import data as datamod
import model

# Contexte chargé à la demande (PAS à l'import de analysis) :
N = MASK = SIS_MODELS = A_SIS = REALVALS = IDX = COLS = None
R2 = config.R2_SEUIL


def _ensure_null():
    """Charge le contexte du modèle nul au premier appel (globals du module)."""
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
    """jours, mc journalier, E_i journalier depuis un tableau (T, N) aligne sur IDX."""
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
    """Nb de courbes R2>seuil par methode (vraies matrices A, integrate_xscan)."""
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
    _ensure_null()
    rng = np.random.default_rng(seed)
    T = REALVALS.shape[0]; nb = T // block
    starts = rng.integers(0, T - block + 1, size=nb)
    out = np.vstack([REALVALS[s:s + block] for s in starts])[:T]
    if out.shape[0] < T:                      # complete a T lignes
        out = np.vstack([out, REALVALS[:T - out.shape[0]]])
    return out


def _one_null(seed):
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


# ═══════════════════════════ Calibration ξ_emp (ex-proxy.py) ═══════════════════════════

"""Calibration de l'énergie de corrélation E_i en probabilité de stress ξ_emp.

Fonctions utilitaires consommées par les notebooks (voir notebooks/article.ipynb).

Le SIS produit x_i(t) ∈ [0, 1] = probabilité que l'actif i soit « infecté ». Pour
le comparer à un observable de **même nature**, on calibre E_i (énergie de
corrélation, ∈ [0, 1] mais pas une probabilité) en probabilité de rendement
extrême, via un indicateur indépendant I_i(t) = 1(|r_i(t)| > q_i^90) :

    ξ_emp(E-bin) = P(I=1 | E ∈ bin) = #{I=1 dans le bin} / #{obs dans le bin}.

`build_mapping` renvoie la fonction f : E ↦ P(stress | E) (isotonique/logistique),
et `map_energy(E, f)` mappe n'importe quel profil E_i(t) en ξ_emp(t) = f(E_i(t)).
"""
from types import SimpleNamespace

import numpy as np


def daily_log_returns(data, all_days):
    """Log-rendements journaliers (N, n_days) alignés sur ``all_days``.

    Les log-rendements intraday 30-min étant additifs, leur somme sur un jour
    donne le log-rendement close-to-close de ce jour.
    """
    df = data.copy()
    df["day"] = df.index.date
    return df.groupby("day").sum().reindex(all_days).values.T


def extreme_indicator(daily_ret, q=0.90):
    """I_i(t) = 1(|r_i(t)| > q_i^q), seuil q **par actif** (NaN où r manquant)."""
    a = np.abs(daily_ret)
    thr = np.nanquantile(a, q, axis=1, keepdims=True)
    I = (a > thr).astype(float)
    I[~np.isfinite(daily_ret)] = np.nan
    return I


def pairs(E, I):
    """Aplati les paires valides (E_i(t), I_i(t)) → deux vecteurs 1D."""
    ok = np.isfinite(E) & np.isfinite(I)
    return E[ok], I[ok].astype(float)


def calibration_curve(Ev, Iv, bin_width=0.05, min_count=50):
    """ξ_emp(E-bin) = P(I=1 | E ∈ bin) par casiers de largeur fixe.

    Returns un dict ``centers, p, n, se`` (se = écart-type binomial), les casiers
    d'effectif < ``min_count`` étant écartés.
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
    """Premier x où y traverse ``level`` (interpolation linéaire), sinon None."""
    d = np.asarray(y) - level
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(s) == 0:
        return None
    i = s[0]
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))


def build_mapping(ctx, q=0.90, kind="isotonic"):
    """Indicateur + paires + courbe + fonctions de mapping E ↦ P(stress|E).

    Returns un ``SimpleNamespace`` : ``f`` (E→proba, selon ``kind``), ``iso`` /
    ``log`` (les deux mappings), ``curve``, ``Ev`` / ``Iv`` (paires),
    ``marginal`` (P(I=1) global), ``q``.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    E = ctx.E_daily_all
    R = daily_log_returns(ctx.data, ctx.all_days)
    I = extreme_indicator(R, q)
    Ev, Iv = pairs(E, I)

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(Ev, Iv)
    lr = LogisticRegression().fit(Ev.reshape(-1, 1), Iv.astype(int))
    iso = lambda e: ir.predict(np.ravel(e))
    log = lambda e: lr.predict_proba(np.ravel(e).reshape(-1, 1))[:, 1]

    return SimpleNamespace(f=iso if kind == "isotonic" else log, iso=iso, log=log,
                           curve=calibration_curve(Ev, Iv), Ev=Ev, Iv=Iv,
                           marginal=float(Iv.mean()), q=q)


def map_energy(E, f):
    """Applique la calibration f à un tableau E (forme et NaN préservés)."""
    E = np.asarray(E, dtype=float)
    out = np.full(E.shape, np.nan)
    m = np.isfinite(E)
    if m.any():
        out[m] = f(E[m])
    return out


# ═══════════════════════════ Orchestration (ex-pipeline.py) ═══════════════════════════

"""Orchestration : enchaîne données → périodes → SIS → sélection.

Couche fine et sans tracé, partagée par tous les scripts ``run_*.py``. Chaque
fonction renvoie des objets déjà mis en cache par les modules sous-jacents, donc
les appeler plusieurs fois (depuis des scripts différents) ne recalcule rien.
"""
import numpy as np
from scipy.stats import linregress

import config
import model
from config import disk_cache, sig_of
from networks import build_filtered_corr, fit_var, prepare_for_sis


def load_periods(ctx):
    """Boucle 1 (fenêtres globales) + boucle 2 (carte de crise par actif, |C|).

    Returns
    -------
    dict
        ``intervals_chrono, mc_smooth, crisis_map``.
    """
    intervals_chrono, mc_smooth = model.detect_global_windows(ctx.mc_all, ctx.mean_mc)
    crisis_map = model.compute_crisis_map(ctx.E_daily_all)
    return dict(intervals_chrono=intervals_chrono, mc_smooth=mc_smooth, crisis_map=crisis_map)


def fit_main(ctx, intervals_chrono, crisis_map):
    """Fits SIS sur les fenêtres croissantes (|C|) + périodes sélectionnées.

    Returns
    -------
    dict
        ``period_data`` (cache) et ``selected`` (périodes avec >=1 fit R²>seuil).
    """
    fc = model.make_fit_context(ctx, crisis_map)
    sig = model.sig_period_data(ctx.N, intervals_chrono, crisis_map)
    period_data = model.fit_periods("period_data", sig, intervals_chrono, fc)
    selected = select_periods(intervals_chrono, period_data)
    return dict(period_data=period_data, selected=selected)


def fit_peaks(ctx, crisis_map):
    """Détection par pics + montées + fits SIS sur les montées.

    Returns
    -------
    dict
        ``peak`` (sortie de detect_peaks), ``peak_rises``, ``peak_pd`` (cache).
    """
    peak = model.detect_peaks(ctx.mc_all, ctx.mean_mc)
    pr = model.peak_rises(peak["peak_intervals"], peak["mc_s"])
    fc = model.make_fit_context(ctx, crisis_map)
    sig = model.sig_peak_period_data(pr, crisis_map)
    peak_pd = model.fit_periods("peak_period_data", sig, pr, fc)
    return dict(peak=peak, peak_rises=pr, peak_pd=peak_pd)


def fit_signed(ctx):
    """Pipeline « sans valeur absolue » : détection signée + fits SIS.

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
    """Diagnostic spectral λ_max/⟨λ⟩ + pics + périodes 'mix' (mis en cache)."""
    diag = model.spectral_diagnostics(ctx.data, ctx.N)
    return dict(diag=diag, lam_dates=model.lambda_peaks(diag),
                mix_periods=model.mix_periods(diag))


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
                xt = model.integrate_xscan(d, gi, A, B=B, R=R)   # intégrateur léger (balayage)
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
            xt = model.integrate_xscan(cache[gi], gi, A_model)
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
