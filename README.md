# Contagion sur les marchés boursiers

Analyse de la contagion entre actifs financiers : réseaux (PMFG, VAR,
corrélation seuillée), dynamique SIS par période de crise, et comparaison
SIS ↔ énergie de corrélation E_i.

## Installation

```bash
pip install numpy pandas matplotlib seaborn statsmodels tqdm pytest networkx openpyxl scikit-learn
```

## Structure

```
contagion/
├── data/                    # CSV des prix + stock_category.xlsx (secteurs)
├── code/                    # Fonctions utilitaires + pipeline (voir code/README.md)
├── notebooks/               # Notebooks (utilisent code/)
│   ├── synthese.ipynb           # Résultats principaux
│   ├── rapport_figures.ipynb    # Figures du rapport → Fig/rapport/
│   ├── paper_chen2023.ipynb     # Modèle SIS théorique
│   ├── sis_matrices_locales.ipynb
│   └── article.ipynb            # Calibration E_i → proba ξ_emp
├── tests/                   # Tests (pytest tests/ -v)
├── Fig/rapport/             # Figures du rapport (seules versionnées)
└── README.md
```

Les fonctions réutilisables sont dans [`code/`](code/README.md), réparties en
6 modules : `config`, `data`, `networks`, `model` (périodes + SIS),
`analysis` (sélection, endo/exo, modèle nul, calibration ξ_emp, orchestration),
`plots`. Les notebooks les utilisent ; ils commencent par
`sys.path.insert(0, '../code')` :

- **`synthese.ipynb`** — résultats principaux : clustering par période, signaux
  réseau dans le temps, grille (B, R), test hors-crise, actifs retenus par
  méthode, communautés, score endo/exo.
- **`rapport_figures.ipynb`** — régénère les figures et tableaux du rapport
  (PDF + PNG) dans `Fig/rapport/`.
- **`paper_chen2023.ipynb`** — modèle SIS théorique : PMFG, dynamique, matrice
  réponse.
- **`article.ipynb`** — calibration de `E_i` en probabilité de stress `ξ_emp`
  (via `analysis`) et comparaison `x_SIS` ↔ `ξ_emp`.

## Lancer le pipeline sans notebook

```bash
python code/run_all.py     # périodes → SIS → courbes → expériences
```

Voir [`code/README.md`](code/README.md) pour les scripts, modules et le cache.

## Tests

```bash
pytest tests/ -v
```
