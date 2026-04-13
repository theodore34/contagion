# Tests Unitaires

Suite de tests pour les fonctions `load_data()` et `correlation()`.

## Exécuter les tests

```bash
pytest tests/test_functions.py -v
```

---

## 📊 Résumé

**Total** : 17 tests | **Réussite** : 100% ✅

---

## TestLoadData (9 tests)

- ✅ Retourne un DataFrame
- ✅ Index défini sur "date"
- ✅ Log-rendements : supprime 1ère ligne (NaN)
- ✅ Log-rendements : formule mathématique correcte
- ✅ Mode prix bruts : préserve les prix
- ✅ Mode prix bruts : conserve toutes les lignes
- ✅ Fusion multi-actifs : contient toutes les colonnes
- ✅ Fusion inner join : garde uniquement dates communes
- ✅ Pas de NaN dans le résultat

## TestCorrelation (8 tests)

- ✅ Dimension matrice : (n_actifs, n_actifs)
- ✅ Dimension matrice avec lag > 0
- ✅ Diagonale = 1.0
- ✅ Matrice symétrique (C = C^T)
- ✅ Valeurs bornées [-1, 1]
- ✅ Correspond à np.corrcoef pour lag=0
- ✅ Corrélation parfaite : séries identiques
- ✅ Cross-corrélation laggée correcte

---

## Installation dépendances

```bash
pip install pytest pandas numpy matplotlib seaborn
```

## Cas de test spécifiques

| Cas | Description |
|-----|-------------|
| Prix doublants | Valide log(2) pour chaque étape |
| Dates non-chevauchées | Fusion inner garde intersection |
| Séries identiques | Corrélation = matrice de 1s |
| Lag cross-corr | Valide corrélation entre séries décalées |
