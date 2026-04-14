# Tests Unitaires

Suite de tests pour `functions.py`.

## Exécuter les tests

```bash
pytest tests/test_functions.py -v
```

---

## Résumé

**Total** : 39 tests | **Réussite** : 100%

---

## TestLoadData (6 tests)

- Log-rendements : supprime 1ère ligne (NaN)
- Log-rendements : formule correcte (log(price_t / price_t-1))
- Mode prix bruts : préserve les valeurs
- Mode prix bruts : garde toutes les lignes
- Fusion inner join : intersection des dates
- Pas de NaN dans résultat

## TestCorrelation (6 tests)

- Diagonale = 1.0
- Matrice symétrique
- Valeurs dans [-1, 1]
- lag=0 correspond à np.corrcoef
- Séries identiques -> matrice de 1s
- Cross-corrélation laggée (lags 1,2,3,5)

## TestCorrThreshold (9 tests)

- Sortie symétrique
- Diagonale préservée
- quantile=0 : zéro le minimum
- Pas de doublement des valeurs
- Exemple connu
- Quantile élevé : zéro la plupart
- diag=False : utilise la matrice complète
- Zéros symétriques
- Pas de mutation de l'entrée

## TestVarContagion (6 tests)

- Dimensions de sortie correctes (1 lag)
- Dimensions de sortie correctes (2 lags)
- Colonnes = noms des actifs
- Coefficients non significatifs mis à zéro
- include_self=False retire les lags propres
- Détecte une dépendance connue (A -> B)

## TestContagionMatrix (4 tests)

- Sortie carrée (N x N)
- Diagonale = 0 (pas d'auto-contagion)
- Index et colonnes cohérents
- Détecte la contagion A -> B

## TestContagionDensity (4 tests)

- Matrice pleine -> densité = 100%
- Matrice vide -> densité = 0%
- Densité partielle calculée correctement
- Actif unique -> densité = 0%

## TestContagionThreshold (4 tests)

- Quantile élevé : zéro les petites valeurs
- quantile=0 : conserve les non-zéros
- Pas de mutation de l'entrée
- Matrice tout-zéro reste tout-zéro

---

## Installation

```bash
pip install pytest pandas numpy statsmodels tqdm
```
