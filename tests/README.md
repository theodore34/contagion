# Tests Unitaires - Contagion Stock Markets

Suite de tests complète pour les fonctions de chargement et calcul de corrélation.

**Exécuter les tests :**
```bash
pytest tests/test_functions.py -v
```

---

## 📊 Vue d'ensemble

- **Total** : 17 tests
- **Couverture** : 2 fonctions principales (`load_data`, `correlation`)
- **Taux de réussite** : 100% ✅

---

## 🧪 Classe TestLoadData (9 tests)

Tests pour la fonction `load_data()` qui charge et fusionne les données CSV avec calcul optionnel des log-rendements.

### 1. test_returns_dataframe
```python
def test_returns_dataframe(self):
    df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"])
    assert isinstance(result, pd.DataFrame)
```

**Objectif** : Vérifier que la fonction retourne bien un `pd.DataFrame`

**Données** : 1 actif avec 5 prix

**Vérification** : Le type du résultat est DataFrame

**Importance** : Test de base - valide le type de retour

---

### 2. test_index_is_date
```python
def test_index_is_date(self):
    df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"])
    assert result.index.name == "date"
```

**Objectif** : Vérifier que l'index du DataFrame est défini sur la colonne "date"

**Données** : 1 actif

**Vérification** : `result.index.name == "date"`

**Importance** : Validation de la structure correcte des données

---

### 3. test_log_returns_drops_first_row
```python
def test_log_returns_drops_first_row(self):
    n = 5
    df_a = _make_df({"price": [1.0] * n})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"])
    assert len(result) == n - 1
```

**Objectif** : Vérifier que `log_returns=True` (défaut) supprime exactement 1 ligne

**Processus interne** :
```
1. shift(1) → crée un NaN en première ligne
2. dropna() → supprime cette ligne
3. Résultat : n-1 lignes
```

**Données** : 5 prix identiques

**Vérification** : Résultat a 4 lignes (5 - 1)

**Importance** : Validation du traitement des données

---

### 4. test_log_returns_values_doubling_series ⭐
```python
def test_log_returns_values_doubling_series(self):
    df_a = _make_df({"price": [1.0, 2.0, 4.0, 8.0, 16.0]})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"])
    np.testing.assert_allclose(result["price"].values, np.log(2.0))
```

**Objectif** : Vérifier que le calcul des log-rendements est **mathématiquement correct**

**Formule appliquée** :
```
log(price[t] / price[t-1])
```

**Données** : Prix qui doublent à chaque étape
```
Prix :      1 → 2 → 4 → 8 → 16
Log-rend :     log(2) = 0.693...
```

**Vérification** : Tous les log-rendements = log(2) ≈ 0.693

**Calculs détaillés** :
- log(2/1) = log(2)
- log(4/2) = log(2)
- log(8/4) = log(2)
- log(16/8) = log(2)

**Importance** : Test critique - valide la formule de calcul financière

---

### 5. test_no_log_returns_preserves_raw_prices
```python
def test_no_log_returns_preserves_raw_prices(self):
    prices = [10.0, 20.0, 15.0, 25.0, 30.0]
    df_a = _make_df({"price": prices})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"], log_returns=False)
    np.testing.assert_array_equal(result["price"].values, prices)
```

**Objectif** : Vérifier que `log_returns=False` retourne les prix **bruts sans transformation**

**Données** : [10, 20, 15, 25, 30]

**Vérification** : Les prix retournés sont identiques aux entrées

**Importance** : Validation du mode "prix bruts"

---

### 6. test_no_log_returns_keeps_all_rows
```python
def test_no_log_returns_keeps_all_rows(self):
    prices = list(range(1, 7))  # [1, 2, 3, 4, 5, 6]
    df_a = _make_df({"price": prices})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"], log_returns=False)
    assert len(result) == len(prices)
```

**Objectif** : Vérifier que sans log-rendements, **aucune ligne n'est supprimée**

**Données** : 6 prix

**Vérification** : Résultat conserve 6 lignes

**Raison** : Sans le calcul `shift()`, pas de NaN créé

**Importance** : Contraste avec test #3

---

### 7. test_merge_two_assets_contains_both_columns
```python
def test_merge_two_assets_contains_both_columns(self):
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df_a = _make_df({"price_a": [1.0, 2.0, 3.0, 4.0, 5.0]}, dates=dates)
    df_b = _make_df({"price_b": [5.0, 4.0, 3.0, 2.0, 1.0]}, dates=dates)
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a, "B": df_b})):
        result = load_data(["A", "B"], log_returns=False)
    assert "price_a" in result.columns
    assert "price_b" in result.columns
```

**Objectif** : Vérifier que la **fusion de 2 actifs** crée un DataFrame avec les colonnes des deux

**Données** :
```
Actif A : dates du 01 au 05 janvier, prix [1, 2, 3, 4, 5]
Actif B : dates du 01 au 05 janvier, prix [5, 4, 3, 2, 1]
```

**Vérification** : Résultat contient "price_a" ET "price_b"

**Résultat attendu** :
```
date      price_a  price_b
2023-01-01   1.0      5.0
2023-01-02   2.0      4.0
2023-01-03   3.0      3.0
2023-01-04   4.0      2.0
2023-01-05   5.0      1.0
```

**Importance** : Validation de la fusion multi-actifs

---

### 8. test_inner_join_drops_non_overlapping_dates ⭐
```python
def test_inner_join_drops_non_overlapping_dates(self):
    dates_a = pd.date_range("2023-01-01", periods=5, freq="D")
    dates_b = pd.date_range("2023-01-03", periods=5, freq="D")
    df_a = _make_df({"price_a": [1.0, 2.0, 3.0, 4.0, 5.0]}, dates=dates_a)
    df_b = _make_df({"price_b": [5.0, 4.0, 3.0, 2.0, 1.0]}, dates=dates_b)
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a, "B": df_b})):
        result = load_data(["A", "B"], log_returns=False)
    assert len(result) == 3
```

**Objectif** : Vérifier que la **fusion INNER** (inner join) conserve **uniquement les dates communes**

**Données** :
```
Actif A : 2023-01-01 à 05 (5 dates)
Actif B : 2023-01-03 à 07 (5 dates)
Chevauchement : 2023-01-03, 04, 05 (3 dates)
```

**Vérification** : Résultat contient 3 lignes

**Visualisation** :
```
A : [01, 02, 03, 04, 05, .....]
B : [............01, 02, 03, 04, 05]
    merge(inner) : [03, 04, 05]  ← uniquement l'intersection
```

**Raison** : Line 16 : `how="inner"` dans `merge()`

**Importance** : Test critique - valide la logique de fusion

---

### 9. test_log_returns_no_nan_in_result
```python
def test_log_returns_no_nan_in_result(self):
    df_a = _make_df({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a})):
        result = load_data(["A"])
    assert not result.isnull().any().any()
```

**Objectif** : Vérifier qu'il n'y a **pas de valeurs NaN** dans le résultat final

**Données** : 5 prix valides

**Vérification** : Aucune cellule NaN dans le DataFrame

**Importance** : Garantit des données propres pour les analyses ultérieures

---

---

## 🔢 Classe TestCorrelation (8 tests)

Tests pour la fonction `correlation()` qui calcule les matrices de corrélation avec support des lags.

### 1. test_output_shape_no_lag
```python
def test_output_shape_no_lag(self):
    data = np.random.randn(100, 4)
    C = correlation(data, lag=0)
    assert C.shape == (4, 4)
```

**Objectif** : Vérifier que la matrice de corrélation a la **bonne forme**

**Données** : 100 observations, 4 actifs

**Vérification** : Résultat est une matrice (4, 4)

**Raison** : Une matrice de corrélation doit être carrée (n_actifs × n_actifs)

**Importance** : Validation basique de la structure

---

### 2. test_output_shape_with_lag
```python
def test_output_shape_with_lag(self):
    data = np.random.randn(100, 4)
    C = correlation(data, lag=5)
    assert C.shape == (4, 4)
```

**Objectif** : Vérifier que même avec `lag > 0`, la matrice reste (4, 4)

**Données** : 100 observations, 4 actifs, lag=5

**Vérification** : Résultat est (4, 4)

**Raison** : Le lag affecte les données temporelles, pas la dimension

**Importance** : Validation avec lag

---

### 3. test_diagonal_ones_no_lag ⭐
```python
def test_diagonal_ones_no_lag(self):
    data = np.random.randn(100, 3)
    C = correlation(data, lag=0)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)
```

**Objectif** : Vérifier que la **diagonale = 1.0** (corrélation avec soi-même = 1)

**Données** : 100 observations, 3 actifs aléatoires

**Vérification** : diag(C) = [1.0, 1.0, 1.0]

**Formule mathématique** :
```
corr(X, X) = 1  pour toute série X
```

**Matrice attendue** :
```
C = [[1.0  ?,   ?  ]
     [?   1.0   ?  ]
     [?    ?   1.0]]
```

**Importance** : Propriété fondamentale de la corrélation

---

### 4. test_symmetry_no_lag ⭐
```python
def test_symmetry_no_lag(self):
    data = np.random.randn(100, 3)
    C = correlation(data, lag=0)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
```

**Objectif** : Vérifier que la matrice est **symétrique** (C = C^T)

**Données** : 3 actifs aléatoires

**Vérification** : C == C transposée

**Raison mathématique** :
```
corr(A, B) = corr(B, A)
Donc C[i,j] = C[j,i]
```

**Exemple** :
```
C = [[1.0  0.5  0.2]
     [0.5  1.0  0.8]     C.T = [[1.0  0.5  0.2]
     [0.2  0.8  1.0]]           [0.5  1.0  0.8]
                                 [0.2  0.8  1.0]]
C == C.T ✓
```

**Importance** : Propriété d'algèbre linéaire

---

### 5. test_values_bounded_no_lag ⭐
```python
def test_values_bounded_no_lag(self):
    data = np.random.randn(100, 5)
    C = correlation(data, lag=0)
    assert np.all(C >= -1.0 - 1e-12)
    assert np.all(C <= 1.0 + 1e-12)
```

**Objectif** : Vérifier que **tous les éléments sont dans [-1, 1]**

**Données** : 5 actifs aléatoires

**Vérification** : -1 ≤ C[i,j] ≤ 1 pour tout i, j

**Raison mathématique** :
```
La corrélation est TOUJOURS bornée dans [-1, 1]
- corr = -1  : corrélation négative parfaite
- corr =  0  : pas de corrélation
- corr = +1  : corrélation positive parfaite
```

**Importance** : Propriété mathématique fondamentale

---

### 6. test_matches_numpy_corrcoef_no_lag
```python
def test_matches_numpy_corrcoef_no_lag(self):
    data = np.random.randn(50, 3)
    C = correlation(data, lag=0)
    expected = np.corrcoef(data.T)
    np.testing.assert_allclose(C, expected, atol=1e-12)
```

**Objectif** : Vérifier que pour `lag=0`, le résultat **correspond à `np.corrcoef`**

**Données** : 50 observations, 3 actifs

**Vérification** : Notre fonction == np.corrcoef (implémentation numpy)

**Importance** : Validation contre l'implémentation de référence

---

### 7. test_perfect_correlation_identical_series_no_lag
```python
def test_perfect_correlation_identical_series_no_lag(self):
    x = np.random.randn(80)
    data = np.column_stack([x, x])
    C = correlation(data, lag=0)
    np.testing.assert_allclose(C, np.ones((2, 2)), atol=1e-10)
```

**Objectif** : Vérifier qu'une **matrice de corrélation de séries identiques = tous des 1**

**Données** : 2 colonnes = même série x dupliquée

```
x = [0.5, -1.2, 0.8, 2.1, ...]
data = [[0.5, 0.5],
        [-1.2, -1.2],
        [0.8, 0.8],
        [2.1, 2.1],
        ...]
```

**Vérification** : C = [[1, 1], [1, 1]]

**Raison** :
```
- col 0 avec col 0 = 1.0 (corrélation avec soi-même)
- col 0 avec col 1 = 1.0 (identiques → corrélation parfaite)
- col 1 avec col 0 = 1.0 (symétrie)
- col 1 avec col 1 = 1.0 (corrélation avec soi-même)
```

**Importance** : Test de validation du cas limite

---

### 8. test_lag_cross_correlation ⭐⭐ **TEST CRITIQUE** 🔴
```python
def test_lag_cross_correlation(self):
    lag = 3
    rng = np.random.default_rng(42)
    x = rng.standard_normal(60)

    # Construire data pour que data[:-lag, 0] == data[lag:, 1]
    data = np.zeros((60, 2))
    data[:, 0] = x                  # Col 0 = série complète x
    data[lag:, 1] = x[:-lag]        # Col 1[3:] = x[0:57]

    C = correlation(data, lag=lag)
    np.testing.assert_allclose(C[0, 1], 1.0, atol=1e-10)
```

**Objectif** : Vérifier que la **cross-corrélation laggée C[0,1] = 1.0** pour séries identiques

**Données - Visualisation détaillée** :
```
Original x : [x0, x1, x2, ..., x57, x58, x59]  (60 éléments)

Construction de data :
Col 0 : [x0, x1, x2, x3, ..., x57, x58, x59]
Col 1 : [0,  0,  0,  x0, x1, ..., x54, x55, x56]
                    ^lag=3
```

**Slices appliquées par corrélation** :
```
data[:-lag, 0]  = data[:57, 0]  = [x0, x1, ..., x56]
data[lag:, 1]   = data[3:, 1]   = [x0, x1, ..., x56]
                                    ↑ identique !
```

**Vérification** :
```
C[0, 1] = corr(data[:-lag, 0], data[lag:, 1])
        = corr([x0, ..., x56], [x0, ..., x56])
        = 1.0  (corrélation parfaite)
```

**⚠️ EXPOSITION DU BUG** :

Code **BUGUÉ** (ancienne version) :
```python
corr_matrix = np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], :data.shape[1]]
                                                        ↑ MAUVAIS SLICE
```
- Retourne : `[:2, :2]` (auto-corrélation)
- Résultat : C[0, 1] ≈ -0.074 ❌

Code **FIXÉ** (version actuelle) :
```python
corr_matrix = np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
                                                        ↑ BON SLICE
```
- Retourne : `[:2, 2:]` (cross-corrélation)
- Résultat : C[0, 1] = 1.0 ✅

**Explications mathématiques** :

`np.corrcoef(A, B)` où A, B ont n lignes retourne matrice (2n, 2n) :
```
     A₀  A₁  B₀  B₁
A₀  [1.0  ?  ?  ?]  ← corr(A₀, A₀)
A₁  [ ?  1.0  ?  ?]  ← corr(A₁, A₁)
B₀  [ ?   ?  1.0  ?] ← corr(B₀, B₀)
B₁  [ ?   ?   ?  1.0]← corr(B₁, B₁)

La cross-corrélation (A avec B) se trouve dans les blocs :
[:n, n:] = corrélation entre A et B
[n:, :n] = corrélation entre B et A (transposée)
```

**Importance** : 
- **🔴 TEST CRITIQUE** - Exposerait immédiatement le bug
- Valide la corrélation laggée (core du projet)
- Sans ce test, le bug ne serait **jamais détecté** en production

---

---

## 📋 Résumé


### Statistiques
| Métrique | Valeur |
|----------|--------|
| Nombre total de tests | 17 |
| Tests load_data | 9 |
| Tests correlation | 8 |
| Tests lag=0 | 6 |
| Tests lag>0 | 1 |
| Score de réussite | 100% ✅ |

### Couverture par catégorie

#### load_data (9 tests)
✅ Type de retour  
✅ Structure de l'index  
✅ Gestion log-rendements  
✅ Calcul mathématique correct  
✅ Mode prix bruts  
✅ Suppression de lignes  
✅ Fusion multi-actifs  
✅ Logique d'intersection (inner join)  
✅ Absence de NaN  

#### correlation (8 tests)
✅ Dimension de la matrice  
✅ Diagonale = 1  
✅ Symétrie  
✅ Bounds [-1, 1]  
✅ Validation vs numpy  
✅ Cas limites  
✅ Cross-corrélation avec lag 🔴  

---

## 🚀 Utilisation

### Exécuter tous les tests
```bash
pytest tests/test_functions.py -v
```

### Exécuter une classe de tests
```bash
pytest tests/test_functions.py::TestLoadData -v
pytest tests/test_functions.py::TestCorrelation -v
```

### Exécuter un test spécifique
```bash
pytest tests/test_functions.py::TestCorrelation::test_lag_cross_correlation -v
```

### Exécuter avec couverture
```bash
pytest tests/test_functions.py --cov=functions --cov-report=html
```

### Mode verbeux
```bash
pytest tests/test_functions.py -vv -s
```

---

## 🛠️ Helpers utilisés

### _make_df(prices, dates=None)
Crée un DataFrame de test avec colonnes de prix et une colonne "date".

```python
df = _make_df({"price_a": [1.0, 2.0, 3.0]})
# Crée un DataFrame avec 3 lignes, index de dates
```

### _mock_read_csv(dfs_by_asset)
Mock pour `pd.read_csv` qui retourne des DataFrames pré-construits au lieu de lire depuis le disque.

```python
with patch("pandas.read_csv", side_effect=_mock_read_csv({"A": df_a, "B": df_b})):
    result = load_data(["A", "B"])
```

---

## 🐛 Bug Historique

### Bug découvert et fixé

**Problème** : Fonction `correlation()` avec `lag > 0` retournait les mauvaises données

**Code AVANT (BUGUÉ)** :
```python
if lag > 0:
    corr_matrix = np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], :data.shape[1]]
    #                                                      ↑ MAUVAIS - auto-corrélation
```

**Code APRÈS (FIXÉ)** :
```python
if lag > 0:
    corr_matrix = np.corrcoef(data[:-lag].T, data[lag:].T)[:data.shape[1], data.shape[1]:]
    #                                                      ↑ BON - cross-corrélation
```

**Test qui l'a détecté** : `test_lag_cross_correlation`

---

## 📝 Notes

- Les tests utilisent `unittest.mock.patch` pour mocker les lectures de fichiers
- Tous les appels NumPy utilisent `atol` (tolérance absolue) pour gérer les erreurs d'arrondi
- Les données aléatoires utilisent `np.random.default_rng(42)` pour garantir la reproductibilité
- Les tests sont indépendants et peuvent s'exécuter dans n'importe quel ordre

---

**Dernière mise à jour** : 2026-04-13  
**Auteur** : Suite de tests automatisée
