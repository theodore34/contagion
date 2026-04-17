# Tests unitaires

Suite minimale de tests pour `functions.py`.

## Exécution

```bash
pytest tests/test_functions.py -v
```

## Contenu

**17 tests**, ciblés sur la forme des sorties et la récupération de coefficients connus sur des séries VAR(1) synthétiques.

| Classe | Tests |
|---|---|
| `TestLoadData` | log-rendements d'une série doublante, inner join sur dates non-overlapping |
| `TestCorrelation` | diagonale à 1 + symétrie, séries identiques, cross-corrélation laggée |
| `TestCorrThreshold` | exemple calculé à la main, symétrie préservée, pas de mutation |
| `TestVarContagion` | forme (N+1, N), récupération de coefficient connu |
| `TestVarContagionMasked` | forme + ligne `const`, récupération coef connu, masque à 0 |
| `TestContagionR2` | R² > 0 sur actif prédictible, agrégation par catégorie |
| `TestRollingContagion` | nombre de fenêtres + shape des matrices, roundtrip cache pkl |

## Dépendances

```bash
pip install pytest pandas numpy statsmodels tqdm
```
