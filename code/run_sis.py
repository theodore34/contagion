#!/usr/bin/env python3
"""Script « SIS » : résout la dynamique SIS en prenant les matrices de contagion.

Pour chaque période détectée et chaque réseau (PMFG, VAR(1), VAR(13), Corr thr),
intègre x_i(t) à partir de x0 = E_i(jour 0), recadre sur le temps de convergence,
et le régresse contre le E_i empirique -> R². Trois jeux de fenêtres :
  - fenêtres croissantes (|C|)  -> cache 'period_data'
  - montées des pics            -> cache 'peak_period_data'
  - détection signée            -> cache 'period_data_sig'

Affiche un récapitulatif des R² par méthode (R² moyen, nb de fits R²>seuil).
Tout est mis en cache (cache/).
"""
import numpy as np

import config
import data
import pipeline


def _summary(period_data, windows, label):
    """Imprime, par méthode, le R² moyen et le nb de fits R²>seuil sur ``windows``."""
    print(f"\n── {label} : R² par methode (sur {len(windows)} fenetres) ──")
    for m in config.SIS_MODELS:
        r2s = [sl["r2"] for (pa, pb) in windows
               for sl in period_data[(pa, pb)][1][m][1].values()]
        if not r2s:
            print(f"  {m:10s} : aucun fit")
            continue
        r2s = np.array(r2s)
        print(f"  {m:10s} : R2_moyen={r2s.mean():.3f}  "
              f"nb fits R2>{config.R2_SEUIL}={int((r2s > config.R2_SEUIL).sum()):3d} / {len(r2s)}")


def main():
    ctx = data.build_context()
    base = pipeline.load_periods(ctx)
    intervals_chrono, crisis_map = base["intervals_chrono"], base["crisis_map"]

    # 1) fenêtres croissantes (|C|)
    main_res = pipeline.fit_main(ctx, intervals_chrono, crisis_map)
    period_data, selected = main_res["period_data"], main_res["selected"]
    _summary(period_data, intervals_chrono, "Fenetres croissantes (|C|)")
    print(f"\n{len(selected)} periode(s) retenue(s) (>=1 fit R²>{config.R2_SEUIL}) : "
          f"{[(str(ctx.all_days[a]), str(ctx.all_days[b])) for a, b in selected]}")

    # 2) montées des pics
    pk = pipeline.fit_peaks(ctx, crisis_map)
    _summary(pk["peak_pd"], pk["peak_rises"], "Montees des pics")

    # 3) détection signée (sans valeur absolue)
    sig = pipeline.fit_signed(ctx)
    _summary(sig["period_data_sig"], sig["intervals_signed"], "Detection signee")
    print(f"\n{len(sig['selected_sig'])} periode(s) signee(s) retenue(s).")


if __name__ == "__main__":
    main()
