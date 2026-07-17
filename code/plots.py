"""Figures du pipeline — la brique « un qui trace les graphiques ».

Backend non interactif (Agg) : chaque fonction construit une figure et la sauve
dans ``Fig/`` (PNG), puis la ferme. Aucune fenêtre n'est ouverte, les scripts
sont donc exécutables sans affichage.

Figures du pipeline : sélection des périodes, comparaison signé/|C|, « 3 vues
par méthode », balayage du seuil q, détection par pics, comparaison des périodes
retenues, superposition λ_max/⟨λ⟩, et balayage des lags VAR.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.stats import linregress

import config

plt.rcParams["figure.dpi"] = 90


def _save(fig, name):
    """Sauve ``fig`` dans ``Fig/{name}.png`` (tight), la ferme et renvoie le chemin."""
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


def _med(series):
    """Médiane par jour relatif (padding NaN à droite) d'une liste de courbes."""
    if not series:
        return np.array([])
    L = max(len(v) for v in series)
    M = np.full((len(series), L), np.nan)
    for r, v in enumerate(series):
        M[r, :len(v)] = v
    return np.nanmedian(M, axis=0)


def _date_ticks(ax, all_days, n=14):
    """Place ~``n`` étiquettes de dates à partir d'un index entier de jours."""
    step = max(1, len(all_days) // n)
    ax.set_xticks(range(0, len(all_days), step))
    ax.set_xticklabels([str(all_days[k]) for k in range(0, len(all_days), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0, len(all_days) - 1)


# ── Étape 1/2 : sélection des périodes (cell 6) ───────────────────────────────
def plot_period_selection(mc_all, mc_smooth, all_days, intervals_chrono,
                          selected, mean_mc, name="periodes_selection"):
    """Série ⟨|C|⟩ + fenêtres de crise (orange) et périodes retenues (rouge)."""
    sel_set = set(selected)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mc_all, color="steelblue", lw=0.7, alpha=0.5, label="<|C_ij|> journalier")
    ax.plot(mc_smooth, color="navy", lw=1.4, label=f"lisse (smooth={config.SMOOTH})")
    ax.axhline(config.CRISIS_FACTOR * mean_mc, color="gray", ls="--", lw=1,
               label=f"seuil crise ({config.CRISIS_FACTOR} x moyenne)")
    for (a, b) in intervals_chrono:
        is_sel = (a, b) in sel_set
        ax.axvspan(a, b, color="red" if is_sel else "orange",
                   alpha=0.30 if is_sel else 0.13)
        if is_sel:
            ax.axvline(a, color="red", lw=0.8, alpha=0.6)
            ax.axvline(b, color="red", lw=0.8, alpha=0.6)
    ax.axvspan(0, 0, color="orange", alpha=0.30,
               label=f"Etape 1 : crise ({len(intervals_chrono)} fenetres)")
    if sel_set:
        ax.axvspan(0, 0, color="red", alpha=0.40,
                   label=f"Etape 2 : selectionnees ({len(sel_set)})")
    _date_ticks(ax, all_days)
    ax.set_ylabel("<|C_ij|>"); ax.grid(alpha=0.3)
    ax.set_title("Selection des periodes (etape 1 crise -> etape 2 R2)")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, name)


# ── Comparaison |C| vs signé (cell 12) ────────────────────────────────────────
def plot_signed_comparison(mc_all, mc_smooth, intervals_chrono, crisis_map, mean_mc,
                           mc_signed, mc_signed_smooth, intervals_signed,
                           crisis_map_signed, all_days, name="periodes_signe_vs_abs"):
    """Deux panneaux : détection sur |C_ij| (haut) et sur C_ij signé (bas)."""
    mean_mc_signed = np.nanmean(mc_signed)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    panels = [(mc_all, mc_smooth, intervals_chrono, crisis_map, mean_mc,
               "|C_ij|  (pipeline actuel)"),
              (mc_signed, mc_signed_smooth, intervals_signed, crisis_map_signed,
               mean_mc_signed, "C_ij signe (sans valeur absolue)")]
    for ax, (mc_v, mc_s, ivs, cm, mv, lab) in zip(axes, panels):
        ax.plot(mc_v, color="steelblue", lw=0.7, alpha=0.5, label="moyenne journaliere")
        ax.plot(mc_s, color="navy", lw=1.4, label=f"lisse (smooth={config.SMOOTH})")
        ax.axhline(config.CRISIS_FACTOR * mv, color="gray", ls="--", lw=1,
                   label=f"seuil crise ({config.CRISIS_FACTOR} x moyenne)")
        for (a, b) in ivs:
            ax.axvspan(a, b, color="orange", alpha=0.20)
        ax.set_title(f"{lab} : {len(ivs)} fenetres globales | "
                     f"{int(cm.any(1).sum())} actifs en crise")
        ax.set_ylabel("correlation moyenne"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    _date_ticks(axes[-1], all_days)
    return _save(fig, name)


# ── Les 3 vues par méthode (cells 10 / 14 / 25) ───────────────────────────────
def plot_three_views(collected, name, suptitle=None, label_suffix="",
                     xlabel="jour", r2_seuil=config.R2_SEUIL):
    """Grille 3 x len(SIS_MODELS) : évolution | scatter | distribution des R².

    Parameters
    ----------
    collected : dict
        Sortie de ``selection.collect_by_method`` : par méthode
        ``dict(Es, Xs, E_all, X_all, r2s)``.
    name : str
        Nom de fichier (sans extension).
    suptitle : str or None
        Titre global éventuel.
    label_suffix : str
        Ajouté au titre de chaque colonne (ex. ' [signe]').
    xlabel : str
        Étiquette de l'axe des x du panneau du haut.
    r2_seuil : float
        Seuil affiché dans les titres.
    """
    models = config.SIS_MODELS
    fig, axes = plt.subplots(3, len(models), figsize=(5 * len(models), 11))
    for j, m in enumerate(models):
        c = collected[m]
        Es, Xs, E_all, X_all, r2s = c["Es"], c["Xs"], c["E_all"], c["X_all"], c["r2s"]

        axt = axes[0, j]
        for E_i, x_traj in zip(Es, Xs):
            ok = ~np.isnan(E_i)
            axt.plot(np.arange(len(E_i))[ok], E_i[ok], color="gray", alpha=0.15, lw=0.7)
            axt.plot(np.arange(len(x_traj)), x_traj, color="red", alpha=0.15, lw=0.7)
        med_E, med_X = _med(Es), _med(Xs)
        if len(med_E):
            axt.plot(np.arange(len(med_E)), med_E, color="black", lw=2.6, label="E_i (mediane)")
            axt.plot(np.arange(len(med_X)), med_X, color="darkred", lw=2.6, label="x_i SIS (mediane)")
        axt.axhline(0, color="gray", lw=0.6, ls=":")
        m_lbl = f"{m} (q={config.CORR_THRESHOLD})" if m == "Corr thr" else m
        axt.set_title(rf"{m_lbl}{label_suffix}  (n={len(Es)}, $R^2$>{r2_seuil})", fontsize=11)
        axt.set_xlabel(xlabel); axt.set_ylabel("E / x"); axt.grid(alpha=0.3)
        if Es:
            axt.legend(fontsize=7)

        axs = axes[1, j]
        if E_all:
            Ec, Xc = np.concatenate(E_all), np.concatenate(X_all)
            axs.scatter(Ec, Xc, s=10, color="#1f77b4", alpha=0.35, edgecolors="none")
            f = linregress(Ec, Xc)
            xs = np.linspace(Ec.min(), Ec.max(), 50)
            axs.plot(xs, f.slope * xs + f.intercept, color="red", lw=2,
                     label=rf"a={f.slope:.2f}, $R^2$={f.rvalue ** 2:.2f}")
            axs.legend(fontsize=8)
        axs.axhline(0, color="gray", lw=0.6, ls=":"); axs.axvline(0, color="gray", lw=0.6, ls=":")
        axs.set_xlabel("E_i empirique"); axs.set_ylabel("x_i SIS"); axs.grid(alpha=0.3)

        axd = axes[2, j]
        if r2s:
            axd.hist(r2s, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
            med_r2 = np.median(r2s)
            axd.axvline(med_r2, color="red", ls="--", lw=1.8, label=f"mediane={med_r2:.2f}")
            axd.legend(fontsize=8)
        axd.set_xlabel(r"$R^2$"); axd.set_ylabel("nb fits"); axd.grid(alpha=0.3)

    if suptitle:
        plt.suptitle(suptitle, fontsize=13)
    return _save(fig, name)


# ── Balayage du seuil q (cells 17 / 18 / 19) ──────────────────────────────────
def plot_qsweep_summary(sweep, name="qsweep_resume", r2_seuil=config.R2_SEUIL):
    """Nb de fits / actifs distincts (R² > seuil) en fonction du quantile q."""
    qs = config.Q_SWEEP
    n_fits, n_assets = [], []
    for q in qs:
        *_, r2s, keys = sweep[round(float(q), 2)]
        good = [k for k, r in zip(keys, r2s) if r > r2_seuil]
        n_fits.append(len(good))
        n_assets.append(len({gi for (_, _, gi) in good}))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qs, n_fits, "o-", color="#1f77b4", label=rf"nb fits ($R^2$>{r2_seuil})")
    ax.plot(qs, n_assets, "s--", color="#d62728", label="nb actifs distincts")
    ax.axvline(config.CORR_THRESHOLD, color="gray", ls=":", lw=1.2,
               label=f"q ref = {config.CORR_THRESHOLD}")
    ax.set_xlabel("seuil q de |C| (quantile)"); ax.set_ylabel(rf"nb ($R^2$ > {r2_seuil})")
    ax.set_title("Corr thr (|C|) : nb d'actifs / fits avec $R^2$ > seuil selon q")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_qsweep_panel(sweep, name="qsweep_panneau", r2_seuil=config.R2_SEUIL):
    """Les « 3 vues » mais une colonne par seuil q (grille grossière Q_PANEL)."""
    qs = config.Q_PANEL
    fig, axes = plt.subplots(3, len(qs), figsize=(4.2 * len(qs), 11), squeeze=False)
    for j, q in enumerate(qs):
        Es, Xs, E_all, X_all, r2s, keys = sweep[round(float(q), 2)]
        sel = [i for i, r in enumerate(r2s) if r > r2_seuil]
        Esf, Xsf = [Es[i] for i in sel], [Xs[i] for i in sel]
        Eaf, Xaf, r2f = [E_all[i] for i in sel], [X_all[i] for i in sel], [r2s[i] for i in sel]

        axt = axes[0, j]
        for E_i, x_traj in zip(Esf, Xsf):
            ok = ~np.isnan(E_i)
            axt.plot(np.arange(len(E_i))[ok], E_i[ok], color="gray", alpha=0.15, lw=0.7)
            axt.plot(np.arange(len(x_traj)), x_traj, color="red", alpha=0.15, lw=0.7)
        med_E, med_X = _med(Esf), _med(Xsf)
        if len(med_E):
            axt.plot(np.arange(len(med_E)), med_E, color="black", lw=2.6, label="E_i (mediane)")
            axt.plot(np.arange(len(med_X)), med_X, color="darkred", lw=2.6, label="x_i SIS (mediane)")
        axt.axhline(0, color="gray", lw=0.6, ls=":")
        axt.set_title(rf"q={q:.2f}  (n={len(sel)}, $R^2$>{r2_seuil})", fontsize=10)
        axt.set_xlabel("jour"); axt.set_ylabel("E / x"); axt.grid(alpha=0.3)
        if Esf:
            axt.legend(fontsize=7)

        axs = axes[1, j]
        if Eaf:
            Ec, Xc = np.concatenate(Eaf), np.concatenate(Xaf)
            axs.scatter(Ec, Xc, s=10, color="#1f77b4", alpha=0.35, edgecolors="none")
            f = linregress(Ec, Xc)
            xs = np.linspace(Ec.min(), Ec.max(), 50)
            axs.plot(xs, f.slope * xs + f.intercept, color="red", lw=2,
                     label=rf"a={f.slope:.2f}, $R^2$={f.rvalue ** 2:.2f}")
            axs.legend(fontsize=8)
        axs.axhline(0, color="gray", lw=0.6, ls=":"); axs.axvline(0, color="gray", lw=0.6, ls=":")
        axs.set_xlabel("E_i empirique"); axs.set_ylabel("x_i SIS"); axs.grid(alpha=0.3)

        axd = axes[2, j]
        if r2f:
            axd.hist(r2f, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
            axd.axvline(np.median(r2f), color="red", ls="--", lw=1.8,
                        label=f"mediane={np.median(r2f):.2f}")
            axd.legend(fontsize=8)
        axd.set_xlabel(r"$R^2$"); axd.set_ylabel("nb fits"); axd.grid(alpha=0.3)

    plt.suptitle(rf"Corr thr (|C|) -- 3 vues par seuil q (fits $R^2$ > {r2_seuil})", fontsize=13)
    return _save(fig, name)


def plot_qsweep_medians(sweep, name="qsweep_medianes", r2_seuil=config.R2_SEUIL):
    """Trajectoires médianes x_i SIS pour chaque q, colorées par q (colorbar)."""
    qs = config.Q_SWEEP
    cmap = plt.cm.viridis
    norm = plt.Normalize(qs.min(), qs.max())
    fig, ax = plt.subplots(figsize=(11, 6))
    medE_ref, last_sel, last_Es = None, None, None
    for q in qs:
        Es, Xs, E_all, X_all, r2s, keys = sweep[round(float(q), 2)]
        sel = [i for i, r in enumerate(r2s) if r > r2_seuil]
        if not sel:
            continue
        ax.plot(np.arange(len(_med([Xs[i] for i in sel]))), _med([Xs[i] for i in sel]),
                color=cmap(norm(q)), lw=1.8, alpha=0.9)
        last_sel, last_Es = sel, Es
        if abs(q - config.CORR_THRESHOLD) < 1e-9:
            medE_ref = _med([Es[i] for i in sel])
    if medE_ref is None and last_sel is not None:
        medE_ref = _med([last_Es[i] for i in last_sel])
    if medE_ref is not None:
        ax.plot(np.arange(len(medE_ref)), medE_ref, color="black", lw=2.8, ls="--",
                label=f"E_i empirique (mediane, ref q={config.CORR_THRESHOLD})")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("seuil q de |C|")
    ax.set_xlabel("jour (depuis debut de la montee)"); ax.set_ylabel("x_i SIS / E_i (medianes)")
    ax.set_title(rf"Corr thr (|C|) : medianes x_i SIS par seuil q (fits $R^2$ > {r2_seuil})")
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3)
    return _save(fig, name)


# ── Détection par les pics (cell 21) ──────────────────────────────────────────
def plot_peak_detection(mc_all, peak, all_days, name="pics_detection"):
    """Série + sommets + largeur des pics (fenêtres de crise par pic)."""
    mc_s, peaks, w_h, l_ips, r_ips = (peak["mc_s"], peak["peaks"], peak["w_h"],
                                      peak["l_ips"], peak["r_ips"])
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mc_all, color="steelblue", lw=0.7, alpha=0.5, label="<|C_ij|> journalier")
    ax.plot(mc_s, color="navy", lw=1.4, label=f"lisse (smooth={config.SMOOTH})")
    ax.axhline(np.nanmean(mc_all), color="gray", ls="--", lw=1, label="moyenne globale")
    for a, b, pk in peak["peak_intervals"]:
        ax.axvspan(a, b, color="red", alpha=0.13)
    ax.hlines(w_h, l_ips, r_ips, color="red", lw=2, alpha=0.7)
    ax.plot(peaks, mc_s[peaks], "v", color="red", ms=9, label="pics (sommets de crise)")
    ax.axvspan(0, 0, color="red", alpha=0.20,
               label=f"crises par pics ({len(peak['peak_intervals'])})")
    _date_ticks(ax, all_days)
    ax.set_ylabel("<|C_ij|>"); ax.grid(alpha=0.3)
    ax.set_title("Detection des crises par les pics de la correlation moyenne")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    return _save(fig, name)


def plot_peak_curves(curves, name="pics_courbes", r2_seuil=config.R2_SEUIL):
    """Montées de pics : évolution E_i/x_i (gauche) + scatter agrégé (droite)."""
    Es, Xs, E_all, X_all = curves["Es"], curves["Xs"], curves["E_all"], curves["X_all"]
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(14, 5))
    for E_i, x_traj in zip(Es, Xs):
        ok = ~np.isnan(E_i)
        axt.plot(np.arange(len(E_i))[ok], E_i[ok], color="gray", alpha=0.20, lw=0.8)
        axt.plot(np.arange(len(x_traj)), x_traj, color="red", alpha=0.20, lw=0.8)
    med_E, med_X = _med(Es), _med(Xs)
    if len(med_E):
        axt.plot(np.arange(len(med_E)), med_E, color="black", lw=2.6, label="E_i (mediane)")
        axt.plot(np.arange(len(med_X)), med_X, color="darkred", lw=2.6, label="x_i SIS (mediane)")
    axt.axhline(0, color="gray", lw=0.6, ls=":")
    axt.set_title(f"Montees de pics : courbes R2 > {r2_seuil}  (n={len(Es)})")
    axt.set_xlabel("jour (depuis debut de la montee)"); axt.set_ylabel("E / x")
    axt.grid(alpha=0.3)
    if Es:
        axt.legend(fontsize=8)
    if E_all:
        Ec, Xc = np.concatenate(E_all), np.concatenate(X_all)
        axs.scatter(Ec, Xc, s=10, color="#1f77b4", alpha=0.35, edgecolors="none")
        f = linregress(Ec, Xc)
        xs = np.linspace(Ec.min(), Ec.max(), 50)
        axs.plot(xs, f.slope * xs + f.intercept, color="red", lw=2,
                 label=f"a={f.slope:.2f}, R2={f.rvalue ** 2:.2f}")
        axs.legend(fontsize=9)
    axs.axhline(0, color="gray", lw=0.6, ls=":"); axs.axvline(0, color="gray", lw=0.6, ls=":")
    axs.set_xlabel("E_i empirique"); axs.set_ylabel("x_i SIS")
    axs.set_title(f"Scatter agrege (courbes R2 > {r2_seuil})"); axs.grid(alpha=0.3)
    return _save(fig, name)


# ── Comparaison des périodes retenues + λ_max/⟨λ⟩ (cell 27) ────────────────────
def plot_period_comparison(mc_all, mc_s, all_days, mean_mc, W_crise, W_pics,
                           diag, lam_dates, mix_periods, name="periodes_comparaison",
                           r2_seuil=config.R2_SEUIL):
    """3 panneaux : périodes retenues sans pics | avec pics | λ_max/⟨λ⟩ + max|ρ|."""
    mc_dates = pd.to_datetime(all_days)
    ymax = np.nanmax(mc_s)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    def _panel(ax, W, color, title):
        ax.plot(mc_dates, mc_all, color="steelblue", lw=0.7, alpha=0.45,
                label="⟨|C_ij|⟩ journalier")
        ax.plot(mc_dates, mc_s, color="navy", lw=1.5, label=f"lissé ({config.SMOOTH} j)")
        ax.axhline(mean_mc, color="gray", ls="--", lw=1, label="moyenne globale")
        for w in W:
            ax.axvspan(w["a"], w["b"], color=color, alpha=0.25)
            ax.axvline(w["a"], color=color, lw=0.8, alpha=0.6)
            ax.axvline(w["b"], color=color, lw=0.8, alpha=0.6)
            mid = w["a"] + (w["b"] - w["a"]) / 2
            ax.text(mid, ymax * 0.99, str(w["n_crise"]), ha="center", va="top",
                    fontsize=8, color="navy")
            ax.text(mid, ymax * 0.88, f"({w['n_sup']})", ha="center", va="top", fontsize=7,
                    color="darkred", fontweight="bold" if w["n_sup"] >= 10 else "normal")
        for d in lam_dates:
            ax.axvline(d, color="green", lw=0.9, ls=":", alpha=0.7)
        ax.set_ylabel("⟨|C_ij|⟩"); ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="upper right", ncol=4)

    _panel(axes[0], W_crise, "orange",
           f"SANS pics — {len(W_crise)} periodes retenues ; bleu = actifs en crise, "
           f"(rouge) = fits R²>{r2_seuil} ; ⋮ vert = pics λ")
    _panel(axes[1], W_pics, "red",
           f"AVEC pics — {len(W_pics)} periodes retenues (≥1 fit R²>{r2_seuil})")

    ax = axes[2]
    for (a, b) in mix_periods:
        ax.axvspan(a, b, color="teal", alpha=0.20)
    ax.plot(diag.index, diag["ratio_eig"], color="black", lw=1.3,
            label="λ_max/⟨λ⟩ (mode de marché)")
    for d in lam_dates:
        ax.axvline(d, color="green", lw=0.9, ls=":", alpha=0.7)
    ax.plot(lam_dates, diag["ratio_eig"].loc[lam_dates], "v", color="green", ms=8,
            label="pics λ")
    ax.set_ylabel("λ_max/⟨λ⟩"); ax.grid(alpha=0.25)
    axb = ax.twinx()
    axb.plot(diag.index, diag["ratio_corr"], color="tab:purple", lw=1.1, alpha=0.8,
             label="max|ρ|/⟨|ρ|⟩")
    axb.set_ylabel("max|ρ|/⟨|ρ|⟩", color="tab:purple")
    axb.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("λ_max/⟨λ⟩ + max|ρ|/⟨|ρ|⟩ → periodes 'mix' (teal) ; ▼ = pics λ",
                 fontsize=11, loc="left")
    extra = [Patch(facecolor="teal", alpha=0.4, label="periodes (mix λ + max corr)")]
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2 + extra, l1 + l2 + [p.get_label() for p in extra],
              fontsize=8, loc="upper right", ncol=2)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("date")
    return _save(fig, name)


# ── Superposition ⟨|C|⟩ et λ_max/⟨λ⟩ (cell 33) ────────────────────────────────
def plot_lambda_overlay(mc_all, mc_s, all_days, mean_mc, diag, lam_dates,
                        name="lambda_overlay"):
    """⟨|C_ij|⟩ (axe gauche) et λ_max/⟨λ⟩ (axe droit) sur le même axe temps."""
    import pandas as pd
    mc_dates = pd.to_datetime(all_days)
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(mc_dates, mc_all, color="steelblue", lw=0.7, alpha=0.40, label="⟨|C_ij|⟩ journalier")
    ax.plot(mc_dates, mc_s, color="navy", lw=1.6, label=f"⟨|C_ij|⟩ lissé ({config.SMOOTH} j)")
    ax.axhline(mean_mc, color="gray", ls="--", lw=1, label="moyenne ⟨|C|⟩")
    ax.set_ylabel("⟨|C_ij|⟩", color="navy"); ax.tick_params(axis="y", labelcolor="navy")
    ax.grid(alpha=0.25)
    axb = ax.twinx()
    axb.plot(diag.index, diag["ratio_eig"], color="crimson", lw=1.4, label="λ_max/⟨λ⟩")
    axb.plot(lam_dates, diag["ratio_eig"].loc[lam_dates], "v", color="crimson", ms=7,
             label="pics λ")
    axb.set_ylabel("λ_max/⟨λ⟩", color="crimson"); axb.tick_params(axis="y", labelcolor="crimson")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_xlim(mc_dates[0], mc_dates[-1])
    ax.set_title("Superposition : ⟨|C_ij|⟩ (bleu) et λ_max/⟨λ⟩ (rouge)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    return _save(fig, name)


# ── Balayage des taux SIS (B, R) ──────────────────────────────────────────────
def plot_br_scan(scan, B_grid, R_grid, metric="n_assets", name="br_scan"):
    """Carte 2D (B vs R) du nb d'actifs/fits retenus, une sous-figure par méthode.

    Parameters
    ----------
    scan : dict
        ``{(B, R): {methode: dict(n_fits, n_assets)}}`` (sortie de run_br_scan).
    B_grid, R_grid : array-like
        Valeurs de B (axe vertical) et R (axe horizontal).
    metric : {'n_assets', 'n_fits'}
        Quantité affichée.
    name : str
        Nom de fichier.
    """
    Bv = [round(float(b), 2) for b in B_grid]
    Rv = [round(float(r), 2) for r in R_grid]
    label = f"nb actifs (R²>{config.R2_SEUIL})" if metric == "n_assets" else f"nb fits (R²>{config.R2_SEUIL})"

    mats = {}
    for m in config.SIS_MODELS:
        M = np.array([[scan[(b, r)][m][metric] for r in Rv] for b in Bv], dtype=float)
        mats[m] = M
    vmin = min(M.min() for M in mats.values())
    vmax = max(M.max() for M in mats.values())

    fig, axes = plt.subplots(1, len(config.SIS_MODELS), figsize=(4.6 * len(config.SIS_MODELS), 4.6))
    im = None
    for ax, m in zip(np.atleast_1d(axes), config.SIS_MODELS):
        M = mats[m]
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Rv))); ax.set_xticklabels([f"{r:.2f}" for r in Rv], fontsize=8)
        ax.set_yticks(range(len(Bv))); ax.set_yticklabels([f"{b:.2f}" for b in Bv], fontsize=8)
        ax.set_xlabel("R (infection)"); ax.set_ylabel("B (recuperation)")
        for i in range(len(Bv)):
            for j in range(len(Rv)):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] < (vmin + vmax) / 2 else "black", fontsize=8)
        if 1.0 in Bv and 1.0 in Rv:                     # cadre rouge = référence B=R=1
            ax.add_patch(Rectangle((Rv.index(1.0) - 0.5, Bv.index(1.0) - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))
        bi, rj = np.unravel_index(np.argmax(M), M.shape)
        ax.set_title(f"{m}\nmax={int(M.max())} @ B={Bv[bi]:.2f}, R={Rv[rj]:.2f}", fontsize=9)
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.025, pad=0.02).set_label(label)
    fig.suptitle(f"Balayage (B, R) : {label} par methode (cadre rouge = ref B=R=1)", fontsize=12)
    return _save(fig, name)


# ── Balayage des lags VAR (cells 35-37) ───────────────────────────────────────
def plot_lag_scan(ldf, lags, name="var_lag_scan", r2_seuil=config.R2_SEUIL):
    """Qualité du fit SIS selon le lag VAR : nb fits R²>seuil + R² agrégé/moyen."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ldf.index, ldf["n_sup"], "o-", color="#4c78a8", label=f"nb fits R²>{r2_seuil}")
    ax.set_xlabel("lag VAR (jours)")
    ax.set_ylabel(f"nb fits R²>{r2_seuil}", color="#4c78a8")
    ax.tick_params(axis="y", labelcolor="#4c78a8")
    ax.set_xticks(lags[::2] if len(lags) > 15 else lags); ax.grid(alpha=0.3)
    axb = ax.twinx()
    axb.plot(ldf.index, ldf["R2_agrege"], "s--", color="#e45756", label="R² agrege")
    axb.plot(ldf.index, ldf["R2_moyen"], "^:", color="darkred", alpha=0.7, label="R² moyen")
    axb.set_ylabel("R²", color="#e45756"); axb.tick_params(axis="y", labelcolor="#e45756")
    for L in (1, 13):
        ax.axvline(L, color="gray", ls=":", lw=1)
    ax.set_title("VAR intermediaires : qualite du fit SIS selon le lag")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    return _save(fig, name)
