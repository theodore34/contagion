# Tests Unitaires

Suite minimale de tests essentiels pour `load_data()` et `correlation()`.

## Exécuter les tests

```bash
pytest tests/test_functions.py -v
```

---

## 📊 Résumé

**Total** : 12 tests | **Réussite** : 100% ✅

---

## TestLoadData (6 tests)

- ✅ Log-rendements : supprime 1ère ligne (NaN)
- ✅ Log-rendements : formule correcte (log(price_t / price_t-1))
- ✅ Mode prix bruts : préserve les valeurs
- ✅ Mode prix bruts : garde toutes les lignes
- ✅ Fusion inner join : intersection des dates
- ✅ Pas de NaN dans résultat

## TestCorrelation (6 tests)

- ✅ Diagonale = 1.0
- ✅ Matrice symétrique
- ✅ Valeurs dans [-1, 1]
- ✅ lag=0 correspond à np.corrcoef
- ✅ Séries identiques → matrice de 1s
- ✅ Cross-corrélation laggée (lags 1,2,3,5)

---

## Installation

```bash
pip install pytest pandas numpy
```
