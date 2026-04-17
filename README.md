# Contagion sur les marchés boursiers

Analyse de la contagion entre actifs financiers par modèles VAR (Vector Autoregression) avec masque de corrélation, fenêtres roulantes et étude de récurrence des liens.

## Installation

```bash
pip install numpy pandas matplotlib seaborn statsmodels tqdm pytest networkx openpyxl
```

## Structure du projet

```
contagion_stock_markets/
├── data/                  # CSV des prix + stock_category.xlsx (secteurs)
├── results/               # Cache pickle des rolling_contagion (indexé par asset_type/q/lag/n)
├── functions.py           # Pipeline complet (utilisé par le notebook)
├── contagion.py           # Version standalone (un seul appel, à partir de CSV)
├── notebook.ipynb         # Notebook d'exploration
└── README.md
```

## `contagion.py` — pipeline standalone

Script autonome qui enchaîne **chargement CSV → log-rendements → masque de corrélation → VAR par fenêtre roulante → cache pickle**. Conçu pour être utilisé sans dépendre du reste de `functions.py` ni du notebook.

### Fonction publique

- **`rolling_contagion(csv_path, corr_quantile, asset_type, interval_size=None, obs_per_regressor=2, lag=1, cache_dir="results", date_col="date")`**

  `csv_path` accepte un chemin unique **ou une liste** de chemins (inner-join sur la colonne `date`). Retourne un dict avec `matrices` (une `DataFrame (N+1, N)` par fenêtre, ligne 0 = `const`), `r2_per_asset`, `r2_total`, `intervals`, `corr_quantile`, `interval_size`, `k_max`, `corr`.

### Helpers internes

- `_load_log_returns(csv_paths, date_col)` : lit un ou plusieurs CSV, inner-join sur la date, calcule les log-rendements.
- `_corr_threshold(corr, quantile)` : met à zéro les entrées dont `|corr|` est sous le quantile.
- `_var_contagion_masked(data, lag, mask)` : VAR équation par équation (OLS + constante), restreint aux régresseurs autorisés par le masque.

### Comportement

- Le masque de corrélation est calculé **une seule fois** sur toutes les données, puis réutilisé pour chaque fenêtre.
- `interval_size` correspond au nombre d'observations `(X, y)` utilisées pour le fit ; la fenêtre brute couvre `interval_size + lag` lignes.
- Si `interval_size=None`, auto-calcul à `obs_per_regressor * k_max`.
- Le résultat est mis en cache dans `results/{asset_type}_q{q}_lag{lag}_n{n}.pkl` ; le second appel recharge le pickle.

### Exemple

```python
from contagion import rolling_contagion

# Un seul actif type
res = rolling_contagion("data/stock_filled.csv", corr_quantile=0.8,
                        asset_type="stock", interval_size=30, lag=1)

# Mélange stocks + crypto (inner-join sur la date)
res = rolling_contagion(["data/stock_filled.csv", "data/crypto_filled.csv"],
                        corr_quantile=0.8, asset_type="stocks-crypto",
                        interval_size=100, lag=2)

print(res["r2_total"])
```

## `functions.py` — pipeline étendu (notebook)

Surcouche utilisée par le notebook. Fonctions principales :

- **`load_data(assets, log_returns=True, sort_by_sector=True)`** : charge et fusionne les CSV par catégorie (`stock`, `crypto`, `etfs`, `indices`), tri optionnel par secteur.
- **`load_categories(path)`** : mapping `asset → secteur` depuis `stock_category.xlsx`.
- **`correlation(data, lag=0)`** : corrélation, avec lag optionnel.
- **`corr_threshold(corr, quantile)`** : masque binarisé par quantile en valeur absolue.
- **`var_contagion(data, n_lags=1)`** / **`var_contagion_masked(data, lag, corr_quantile=None, mask=None)`** : VAR sans / avec masque de corrélation.
- **`contagion_r2(data, matrix, lag, categories=None)`** : R² global, par actif, et agrégé par secteur.
- **`rolling_contagion(data, corr_quantile, asset_type, interval_size=None, lag=1, ...)`** : variante qui prend un `DataFrame` déjà chargé en mémoire (même format de retour que `contagion.py`).
- **`activation_frequency(data, corr_quantile, asset_type, interval_size=None, lag=1, binarization_quantile=0.8, eps=1e-6, plot=True)`** : binarise chaque matrice roulante (seuil = `binarization_quantile` des coefficients non nuls) puis renvoie la fréquence d'activation par lien. La diagonale est toujours mise à zéro.
- **`mean_magnitude(data, corr_quantile, asset_type, interval_size=None, lag=1, eps=1e-6, plot=True)`** : magnitude signée moyenne des coefficients sur les fenêtres. Diagonale toujours à zéro.

## Cache

`results/` contient les pickles indexés par `{asset_type}_q{corr_quantile}_lag{lag}_n{interval_size}.pkl`. Les appels suivants avec les mêmes paramètres rechargent directement depuis le cache.
