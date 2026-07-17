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
