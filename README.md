# Contagion sur les marchés boursiers

Analyse de la contagion entre actifs financiers par des modèles VAR (Vector Autoregression) et des matrices de corrélation.

## Installation

```bash
pip install numpy pandas matplotlib seaborn statsmodels tqdm pytest
```

## Structure du projet

```
contagion_stock_markets/
├── data/                  # Fichiers CSV des prix (par actif)
├── functions.py           # Fonctions d'analyse
├── notebook.ipynb         # Notebook d'exploration
├── tests/
│   ├── test_functions.py  # Tests unitaires
│   └── README.md          # Détail des tests
└── README.md
```

## Fonctions disponibles

### Chargement des données

- **`load_data(assets, log_returns=True)`** : charge et fusionne les CSV de prix, calcule optionnellement les log-rendements.

### Corrélation

- **`correlation(data, lag=0)`** : matrice de corrélation avec lag optionnel.
- **`corr_threshold(corr, quantile, diag=True)`** : filtre la matrice de corrélation par seuil de quantile.

### Contagion (VAR)

- **`var_contagion(data, n_lags=1, pvalue_threshold=0.1, include_self=True)`** : estime un VAR équation par équation et retourne les coefficients significatifs.
- **`contagion_matrix(data, n_lags=1, pvalue_threshold=0.1, lag_name="L1")`** : extrait une matrice carrée (N x N) de contagion pour un lag donné. La diagonale est mise à zéro.
- **`contagion_density(matrix)`** : calcule la densité du réseau de contagion (% de liens non nuls).
- **`contagion_threshold(matrix, quantile)`** : filtre la matrice de contagion par quantile en valeur absolue.
- **`var_contagion_masked(data, lag=1, corr_quantile=0.5, pvalue_threshold=0.1)`** : VAR à un seul lag, ne fittant que les coefficients non masqués par la corrélation seuillée à lag 0.

## Utilisation rapide

```python
from functions import load_data, contagion_matrix, contagion_density

# Charger les log-rendements
data = load_data(["SP500", "CAC40", "NIKKEI"])

# Matrice de contagion (VAR lag 1, seuil p < 0.1)
matrix = contagion_matrix(data, n_lags=1, pvalue_threshold=0.1)

# Densité du réseau
print(f"Densité : {contagion_density(matrix):.1f}%")
```

## Tests

```bash
pytest tests/test_functions.py -v
```

45 tests couvrant toutes les fonctions. Voir [tests/README.md](tests/README.md) pour le détail.
