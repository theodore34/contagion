# Contagion sur les marchés boursiers

Analyse de la contagion entre actifs financiers : réseaux de contagion
(PMFG, VAR, corrélation seuillée), dynamique SIS par période de crise, et
comparaison SIS ↔ énergie de corrélation E_i.

## Installation

```bash
pip install numpy pandas matplotlib seaborn statsmodels tqdm pytest networkx openpyxl scikit-learn
```

## Structure du projet

```
contagion_stock_markets/
├── data/                    # CSV des prix + stock_category.xlsx (secteurs)
├── code/                    # Package pipeline (voir code/README.md)
├── tests/                   # Tests unitaires du package (pytest tests/ -v)
├── Fig/rapport/             # Figures finales du rapport (seules versionnées)
├── synthese.ipynb           # Résultats principaux (6 étapes, construit sur code/)
├── rapport_figures.ipynb    # Figures HD du rapport de stage → Fig/rapport/
├── paper_chen2023.ipynb     # Reproduction de Chen et al. (2023) : SIS, PMFG, matrice réponse
└── README.md
```

Tout le code réutilisable vit dans le package [`code/`](code/README.md)
(chargement des données, matrices de contagion, détection des périodes,
dynamique SIS, sélection, figures). Les notebooks sont des consommateurs de ce
package :

- **`synthese.ipynb`** — notebook propre des résultats : clustering par
  période, signaux réseau dans le temps, grille (B, R), test hors-crise,
  actifs retenus par méthode, communautés de Louvain, score endo/exo.
- **`rapport_figures.ipynb`** — régénère les figures et tableaux du rapport
  (PDF + PNG 300 dpi) dans `Fig/rapport/`.
- **`paper_chen2023.ipynb`** — référence théorique : dérivation complète du
  modèle (PMFG, champ moyen SIS, jacobien, matrice réponse G, temps de
  réponse τ).

## Lancer le pipeline sans notebook

```bash
python code/run_all.py     # périodes → SIS → courbes → expériences
```

Voir [`code/README.md`](code/README.md) pour le détail des scripts, des
modules et du cache.

## Tests

```bash
pytest tests/ -v
```
