#!/usr/bin/env python3
"""Script « balayage (B, R) » : les taux SIS donnent-ils de meilleurs résultats ?

Pour chaque couple (B, R) d'une grille (par défaut 0.8 → 1.2 par pas de 0.1),
recompte, sur toutes les fenêtres de crise (|C|) et par méthode (PMFG, VAR(1),
VAR(13), Corr thr), le nombre d'actifs distincts dont le fit SIS atteint
R² > seuil. Les conditions initiales (E_i, x0) sont réutilisées du cache — seule
l'intégration SIS dépend de (B, R) — donc le balayage est rapide et parallélisé.

Figures (Fig/) : carte 2D B vs R du nb d'actifs retenus (et du nb de fits) par
méthode, avec la référence B=R=1 encadrée en rouge.
Cache : 'br_scan'.
"""
import config
import data
import pipeline
import plots


def main():
    ctx = data.build_context()
    base = pipeline.load_periods(ctx)
    intervals_chrono, crisis_map = base["intervals_chrono"], base["crisis_map"]
    period_data = pipeline.fit_main(ctx, intervals_chrono, crisis_map)["period_data"]

    scan = pipeline.run_br_scan(ctx, intervals_chrono, crisis_map, period_data,
                                config.BR_GRID, config.BR_GRID)

    # ── Récapitulatif : référence (B=R=1) vs meilleur (B, R) par méthode ──────
    ref = (1.0, 1.0)
    has_ref = ref in scan
    print(f"\nBalayage (B, R) sur {config.BR_GRID[0]:.1f}..{config.BR_GRID[-1]:.1f} "
          f"({len(config.BR_GRID)}x{len(config.BR_GRID)} = {len(scan)} couples)")
    print(f"{'methode':10s} | {'ref B=R=1':>10s} | {'meilleur':>9s} @ (B, R)")
    print("-" * 52)
    for m in config.SIS_MODELS:
        best_bp, best_n = max(((bp, v[m]["n_assets"]) for bp, v in scan.items()),
                              key=lambda t: t[1])
        ref_n = scan[ref][m]["n_assets"] if has_ref else float("nan")
        gain = f"(+{best_n - ref_n})" if has_ref and best_n > ref_n else "(=)"
        print(f"{m:10s} | {ref_n:10.0f} | {best_n:9.0f} @ B={best_bp[0]:.1f}, R={best_bp[1]:.1f} {gain}")

    plots.plot_br_scan(scan, config.BR_GRID, config.BR_GRID, metric="n_assets")
    plots.plot_br_scan(scan, config.BR_GRID, config.BR_GRID, metric="n_fits",
                       name="br_scan_fits")


if __name__ == "__main__":
    main()
