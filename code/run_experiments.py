#!/usr/bin/env python3
"""Script « expériences » : balayage du seuil q et balayage des lags VAR.

  - Balayage du quantile q de la matrice 'Corr thr' : pour chaque q, refit SIS
    de la seule méthode 'Corr thr' (réseaux PMFG/VAR inchangés), sur tous les
    actifs en crise. Figures : résumé n(q), panneau 3 vues par q, médianes par q.
  - Balayage des lags VAR (1..29) : qualité du fit SIS (nb fits R²>seuil + R²
    agrégé/moyen) -> cherche un éventuel pic intermédiaire entre lag 1 et 13.

Caches : 'qsweep_corrthr', 'var_lag_scan' (cache/).
"""
import config
import data
import pipeline
import plots


def main():
    ctx = data.build_context()
    base = pipeline.load_periods(ctx)
    intervals_chrono, crisis_map = base["intervals_chrono"], base["crisis_map"]
    main_res = pipeline.fit_main(ctx, intervals_chrono, crisis_map)
    period_data, selected = main_res["period_data"], main_res["selected"]

    # ── Balayage du seuil q de 'Corr thr' ─────────────────────────────────────
    sweep = pipeline.run_qsweep(ctx, intervals_chrono, crisis_map, period_data)
    print(f"\nBalayage q : {len(config.Q_SWEEP)} seuils de {config.Q_SWEEP[0]:.2f} "
          f"a {config.Q_SWEEP[-1]:.2f}")
    for q in config.Q_SWEEP:
        *_, r2s, keys = sweep[round(float(q), 2)]
        good = [k for k, r in zip(keys, r2s) if r > config.R2_SEUIL]
        n_assets = len({gi for (_, _, gi) in good})
        print(f"  q={q:.2f} : {len(good):3d} fits R²>{config.R2_SEUIL}  ({n_assets:3d} actifs distincts)")
    plots.plot_qsweep_summary(sweep)
    plots.plot_qsweep_panel(sweep)
    plots.plot_qsweep_medians(sweep)

    # ── Balayage des lags VAR ─────────────────────────────────────────────────
    ldf = pipeline.run_lag_scan(ctx, selected, period_data)
    print("\nBalayage des lags VAR -> qualite du fit SIS :")
    print(ldf.round(3).to_string())
    best_nsup, best_agg = int(ldf["n_sup"].idxmax()), int(ldf["R2_agrege"].idxmax())
    verdict = ("PIC INTERMEDIAIRE (ni 1 ni 13)" if best_agg not in (1, 13)
               else "optimum a une extremite (1 ou 13)")
    print(f"\nmax nb fits R²>{config.R2_SEUIL} au lag {best_nsup} ; "
          f"max R² agrege au lag {best_agg}  ->  {verdict}")
    plots.plot_lag_scan(ldf, config.LAG_SCAN)


if __name__ == "__main__":
    main()
