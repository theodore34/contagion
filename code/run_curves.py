#!/usr/bin/env python3
"""Script « courbes » : examine les fits SIS, sélectionne les courbes, trace tout.

À partir des fits mis en cache (period_data / peak_period_data / period_data_sig) :
  - écrit les tableaux récapitulatifs R² par actif x période (CSV à la racine) ;
  - sélectionne les courbes R² > seuil et les périodes retenues ;
  - trace les « 3 vues par méthode » (|C|, signé, pics), le scatter des montées,
    le panneau de sélection des périodes (étape 1 + étape 2 en rouge) et la
    comparaison des périodes retenues (avec/sans pics) + λ_max/⟨λ⟩.

CSV : results/actifs_crise_R2.csv, results/actifs_montees_pics_R2.csv.
Figures : Fig/.
"""
import config
import data
import analysis
import plots
import analysis


def main():
    ctx = data.build_context()
    base = analysis.load_periods(ctx)
    intervals_chrono, mc_smooth, crisis_map = (base["intervals_chrono"],
                                               base["mc_smooth"], base["crisis_map"])

    # ── Fits (tous depuis le cache) ───────────────────────────────────────────
    main_res = analysis.fit_main(ctx, intervals_chrono, crisis_map)
    period_data, selected = main_res["period_data"], main_res["selected"]
    pk = analysis.fit_peaks(ctx, crisis_map)
    sig = analysis.fit_signed(ctx)
    spec = analysis.spectral(ctx)

    # ── Tableaux R² -> CSV ────────────────────────────────────────────────────
    crise_df = analysis.crisis_table(intervals_chrono, period_data, ctx, crisis_map,
                                      csv_path=config.RESULTS_DIR / "actifs_crise_R2.csv")
    print(f"{len(crise_df)} lignes (actif x periode) -> results/actifs_crise_R2.csv")
    pk_df, peak_curves = analysis.peak_table(pk["peak_rises"], pk["peak_pd"], ctx,
                                              csv_path=config.RESULTS_DIR / "actifs_montees_pics_R2.csv")
    good = pk_df[pk_df["R2_max"] > config.R2_SEUIL]
    print(f"{len(pk_df)} courbes (actif x montee) ; {len(good)} avec R2>{config.R2_SEUIL} "
          f"({good.actif.nunique()} actifs distincts) -> results/actifs_montees_pics_R2.csv")

    # ── Redondance par méthode (sans pics | avec pics) ────────────────────────
    print("\nRedondance des actifs (R²>seuil) — sans pics | avec pics :")
    print(analysis.redundancy_recap(period_data, pk["peak_pd"], ctx).to_string())

    # ── Figures : sélection + 3 vues + scatter + comparaison ──────────────────
    plots.plot_period_selection(ctx.mc_all, mc_smooth, ctx.all_days,
                                intervals_chrono, selected, ctx.mean_mc,
                                name="periodes_selection_retenues")

    plots.plot_three_views(analysis.collect_by_method(period_data, selected),
                           name="courbes_3vues_abs",
                           suptitle="Resultat 2 : 3 vues par methode (|C|)")
    plots.plot_three_views(analysis.collect_by_method(sig["period_data_sig"], sig["selected_sig"]),
                           name="courbes_3vues_signe", label_suffix=" [signe]",
                           suptitle="Les 3 vues avec C_ij signe (detection signee)")
    plots.plot_three_views(analysis.collect_by_method(pk["peak_pd"], pk["peak_rises"]),
                           name="courbes_3vues_pics", xlabel="jour (depuis debut montee)",
                           suptitle="Montees de pics : 3 vues par methode")
    plots.plot_peak_curves(peak_curves)

    # ── Comparaison des périodes retenues (avec/sans pics) + λ_max/⟨λ⟩ ─────────
    W_crise = analysis.retain(analysis.window_summaries(intervals_chrono, period_data, ctx.all_days))
    W_pics = analysis.retain(analysis.window_summaries(pk["peak_rises"], pk["peak_pd"], ctx.all_days))
    plots.plot_period_comparison(ctx.mc_all, pk["peak"]["mc_s"], ctx.all_days, ctx.mean_mc,
                                 W_crise, W_pics, spec["diag"], spec["lam_dates"],
                                 spec["mix_periods"])
    nc_c, ns_c = analysis.totals(W_crise)
    nc_p, ns_p = analysis.totals(W_pics)
    print(f"\nSANS pics : {len(W_crise)} periodes retenues ; {nc_c} actifs en crise ; {ns_c} fits R²>seuil")
    print(f"AVEC pics : {len(W_pics)} periodes retenues ; {nc_p} actifs en crise ; {ns_p} fits R²>seuil")


if __name__ == "__main__":
    main()
