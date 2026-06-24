#!/usr/bin/env python3
"""Pipeline complet de bout en bout (A → Z), tout en cache.

Enchaîne les quatre scripts dans l'ordre :
  1. périodes      (run_periods)
  2. dynamique SIS (run_sis)
  3. courbes + CSV (run_curves)
  4. expériences   (run_experiments)

Le contexte (données + base) n'est chargé qu'une fois ; tous les calculs lourds
sont mis en cache (cache/) donc une seconde exécution est quasi instantanée.
"""
import run_curves
import run_experiments
import run_periods
import run_sis


def main():
    print("=" * 70, "\n[1/4] PERIODES\n", "=" * 70, sep="")
    run_periods.main()
    print("\n", "=" * 70, "\n[2/4] DYNAMIQUE SIS\n", "=" * 70, sep="")
    run_sis.main()
    print("\n", "=" * 70, "\n[3/4] COURBES + SELECTION\n", "=" * 70, sep="")
    run_curves.main()
    print("\n", "=" * 70, "\n[4/4] EXPERIENCES (q-sweep, lag scan)\n", "=" * 70, sep="")
    run_experiments.main()
    print("\nTermine. Figures dans Fig/, caches dans cache/, CSV a la racine.")


if __name__ == "__main__":
    main()
