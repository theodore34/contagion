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
