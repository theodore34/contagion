# Pipeline contagion SIS ↔ E_i

Pipeline A → Z en scripts (refonte historique du notebook
`monthly_correlations.ipynb`, aujourd'hui supprimé — voir l'historique git).
Chaque étape met ses résultats lourds en cache (`cache/`, pickles) et ses
figures dans `Fig/`. Les calculs déjà faits ne sont jamais refaits : les
signatures de cache sont sensibles au contenu (données, fenêtres, paramètres).

## Lancer

Depuis la racine du dépôt (ou depuis `code/`) :

```bash
python code/run_all.py          # tout : périodes → SIS → courbes → expériences
python code/run_periods.py      # détection des périodes seule
python code/run_sis.py          # dynamique SIS (prend les matrices) + R²
python code/run_curves.py       # tableaux R² (CSV) + figures de courbes/sélection
python code/run_experiments.py  # balayage du seuil q + balayage des lags VAR
python code/run_br_scan.py      # balayage des taux SIS (B, R) -> carte 2D par méthode
```

`run_br_scan.py` réutilise les conditions initiales déjà en cache (indépendantes
de B/R) et ne recalcule que l'intégration SIS, puis tout est lu depuis le cache
`br_scan`. Le parallélisme est plafonné par `config.N_JOBS` (borne la RAM).

## Modules (6)

| Fichier         | Rôle |
|-----------------|------|
| `config.py`     | chemins (`cache/`, `Fig/`), **tous** les hyperparamètres, et le cache disque (`disk_cache` / `sig_of`) |
| `networks.py`   | matrices de contagion : PMFG, VAR(d), Corr thr |
| `data.py`       | chargement des rendements + calculs de base (E_i, ⟨\|C\|⟩, réseaux) → cache `base` |
| `model.py`      | détection des périodes de crise + dynamique SIS (intégrateurs, fits par période) |
| `analysis.py`   | sélection (R²), réflexivité endo/exo, modèle nul, calibration ξ_emp, orchestration |
| `plots.py`      | toutes les figures → `Fig/` (backend Agg) |
| `run_*.py`      | scripts exécutables (un par tâche) |

Le balayage (B, R) : `config.BR_GRID` fixe les valeurs de B (récupération) et R
(infection) testées (défaut `[0.8, 1.0, 1.2]`). Pour chaque couple, on
recompte par méthode les actifs distincts (et les fits) avec R² > `R2_SEUIL`,
puis on trace une carte 2D B×R par méthode (référence B=R=1 encadrée).

## Données

`data/stock_filled.csv` (+ `stock_category.xlsx` pour le tri par secteur) :
N = 146 actions, log-rendements intraday 30 min, 2019-04 → 2023-05.

## Cache

`cache/` est amorcé avec les pickles déjà calculés (signatures sensibles au
contenu). Mettre `config.FORCE_RECOMPUTE = True` pour tout recalculer. Supprimer un fichier de `cache/` force le recalcul
de cette seule étape.
