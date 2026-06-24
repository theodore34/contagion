#!/usr/bin/env python3
"""Script « périodes » : détecte et met en cache toutes les périodes de crise.

Produit :
  - les fenêtres croissantes de ⟨|C|⟩ (boucle 1) et la carte de crise par actif ;
  - les crises par pics de ⟨|C|⟩ et leurs montées ;
  - la détection « signée » (sans valeur absolue), pour comparaison ;
  - le diagnostic spectral λ_max/⟨λ⟩.

Figures écrites dans Fig/ : sélection des périodes (étape 1), comparaison
signé/|C|, détection par pics, superposition ⟨|C|⟩ vs λ_max/⟨λ⟩.

Tout est mis en cache (cache/) : relancer ne recalcule que ce qui a changé.
"""
import data
import periods
import pipeline
import plots


def main():
    ctx = data.build_context()

    base = pipeline.load_periods(ctx)
    intervals_chrono = base["intervals_chrono"]
    mc_smooth = base["mc_smooth"]
    crisis_map = base["crisis_map"]

    peak = periods.detect_peaks(ctx.mc_all, ctx.mean_mc)
    peak_rises = periods.peak_rises(peak["peak_intervals"], peak["mc_s"])

    # détection signée seule (pas de SIS ici : ce script ne fait que les périodes)
    E_signed = periods.compute_E_signed(ctx.data, ctx.N)
    mc_signed, mc_signed_smooth, intervals_signed, crisis_map_signed = \
        periods.detect_global_windows_signed(E_signed)
    spec = pipeline.spectral(ctx)

    # ── Résumés chiffrés ──────────────────────────────────────────────────────
    print(f"\nBoucle 1 (|C|) : {len(intervals_chrono)} fenetres de crise")
    for (a, b) in intervals_chrono:
        print(f"  {ctx.all_days[a]} -> {ctx.all_days[b]}  ({b - a + 1} j)")
    print(f"Boucle 2 (|C|) : {int(crisis_map.any(axis=1).sum())} actifs en crise")
    print(f"\nPics de ⟨|C|⟩ : {len(peak['peak_intervals'])} pic(s), "
          f"{len(peak_rises)} montee(s) (creux -> sommet)")
    print(f"\nSigne (sans abs) : {len(intervals_signed)} fenetres ; "
          f"{int(crisis_map_signed.any(1).sum())} actifs en crise")
    print(f"\nSpectral : {len(spec['lam_dates'])} pic(s) de λ_max/⟨λ⟩ ; "
          f"{len(spec['mix_periods'])} periode(s) 'mix'")

    # ── Figures (périodes seules) ─────────────────────────────────────────────
    plots.plot_period_selection(ctx.mc_all, mc_smooth, ctx.all_days,
                                intervals_chrono, [], ctx.mean_mc)
    plots.plot_signed_comparison(ctx.mc_all, mc_smooth, intervals_chrono, crisis_map,
                                 ctx.mean_mc, mc_signed, mc_signed_smooth,
                                 intervals_signed, crisis_map_signed, ctx.all_days)
    plots.plot_peak_detection(ctx.mc_all, peak, ctx.all_days)
    plots.plot_lambda_overlay(ctx.mc_all, peak["mc_s"], ctx.all_days, ctx.mean_mc,
                              spec["diag"], spec["lam_dates"])


if __name__ == "__main__":
    main()
