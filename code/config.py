"""Chemins et hyperparamètres — point unique de réglage de tout le pipeline.

Tous les autres modules importent leurs constantes d'ici. Modifier une valeur
ici invalide automatiquement les caches concernés (les signatures de cache
intègrent ces paramètres), donc il suffit de relancer le script voulu.
"""
from pathlib import Path

# ── Arborescence ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent      # racine du dépôt
CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data"                            # CSV de prix (stock_filled.csv ...)
CACHE_DIR = ROOT / "cache"                          # pickles de résultats lourds
FIG_DIR = ROOT / "Fig"                              # figures produites
RESULTS_DIR = ROOT / "results"                      # exports CSV/figures finales
for _d in (CACHE_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# Recalcule tout en ignorant le cache si True (mettre à True ponctuellement).
FORCE_RECOMPUTE = False

# Plafond de workers joblib pour les balayages parallèles : borne la RAM
# (chaque worker duplique les caches ; trop de workers => explosion mémoire).
N_JOBS = 2

# ── Données ───────────────────────────────────────────────────────────────────
ASSETS = ["stock"]          # fichiers data/{a}_filled.csv fusionnés sur la date
LOG_RETURNS = True
SORT_BY_SECTOR = True

# ── Détection des fenêtres de crise (corrélation / E_i croissants) ────────────
SMOOTH = 14                 # largeur du lissage (moyenne glissante) avant détection
MIN_LEN = 12                # longueur minimale d'une fenêtre croissante (jours)
MERGE_GAP = 3               # fusionne 2 fenêtres séparées par <= ce nb de jours décroissants
REFINE = SMOOTH // 2        # ajustement fin des bords (±REFINE) pour maximiser l'amplitude
CRISIS_FACTOR = 0.6         # fenêtre GLOBALE gardée si <|C|>_fenêtre > CRISIS_FACTOR x moyenne
ASSET_FACTOR = 0.6          # actif "en crise" si E_i_fenêtre > ASSET_FACTOR x sa moyenne

# ── Sous-fenêtre de forte montée par actif (strong_rise) ──────────────────────
SR_SMOOTH = 2               # lissage de la sous-fenêtre
SR_MIN_LEN = 10             # longueur minimale de la sous-fenêtre (jours)
SR_MIN_GAIN = 0.5           # gain (montée) minimal requis sur la sous-fenêtre

# ── Réseau de contagion ───────────────────────────────────────────────────────
CORR_THRESHOLD = 0.85       # quantile de |C| pour la matrice 'Corr thr'
SIS_MODELS = ["PMFG", "VAR(1)", "VAR(13)", "Corr thr"]

# ── Modèle SIS ────────────────────────────────────────────────────────────────
B_FIT, R_FIT = 1.0, 1.0     # taux de récupération B et d'infection R
T_LONG = 50.0               # horizon d'intégration pour trouver l'équilibre x*
TOL_EQ = 0.01               # tolérance sur x* pour le temps de convergence T_conv

# ── Sélection des courbes / périodes ──────────────────────────────────────────
R2_SEUIL = 0.6              # seuil R² par fit pour les courbes / scatter retenus
N_MIN = 20                  # nb minimal d'actifs en crise pour retenir une période
R2_PERIOD = 0.2             # R² moyen minimal (meilleure méthode) pour retenir une période

# ── Détection des crises par les PICS de <|C_ij|> ─────────────────────────────
PK_PROM_FACTOR = 0.30       # proéminence minimale = facteur x écart-type de <|C|>
PK_DISTANCE = MIN_LEN       # écart minimal (jours) entre deux pics
PK_RELHEIGHT = 0.6          # fraction de la proéminence -> largeur du pic

# ── Diagnostic spectral λ_max/⟨λ⟩ (façon article.ipynb) ───────────────────────
WINDOW_SPEC = 13 * 21       # ~1 mois de barres 30-min
STEP_SPEC = 13              # 1 point / jour
SPEC_SMOOTH = 15            # lissage avant détection des pics de λ_max/⟨λ⟩
SPEC_PROM_FACTOR = 0.30     # proéminence des pics de λ_max/⟨λ⟩ (x écart-type)

# ── Expériences ───────────────────────────────────────────────────────────────
import numpy as _np
# balayage du seuil q de 'Corr thr' : pas grossier (0.05) puis fin (0.01) dans la zone du saut
Q_SWEEP = _np.round(_np.concatenate([_np.arange(0.50, 0.80, 0.05),
                                     _np.arange(0.80, 0.96, 0.01)]), 2)
Q_PANEL = _np.round(_np.arange(0.50, 0.96, 0.05), 2)   # grille grossière pour le panneau
LAG_SCAN = list(range(1, 30))                          # lags VAR balayés pour le fit SIS
# balayage des taux SIS : B (récupération) et R (infection), même grille pour les deux
BR_GRID = _np.round(_np.arange(0.80, 1.21, 0.20), 2)   # [0.8, 1.0, 1.2] (réduit : RAM)


# ═══════════════════════════ Cache disque (ex-cache.py) ═══════════════════════════

"""Cache disque minimal : un calcul lourd n'est fait qu'une seule fois.

`disk_cache(name, sig, compute)` charge ``cache/{name}.{sig}.pkl`` s'il existe,
sinon exécute ``compute()``, sauve le résultat et le renvoie. La signature
``sig`` doit changer dès que le résultat changerait : utiliser :func:`sig_of`
pour fabriquer une signature sensible au CONTENU (np.ndarray inclus).

Schéma de nommage stable : les pickles déjà produits sont réutilisés tels quels
d'une exécution à l'autre.
"""
import glob
import hashlib
import os
import pickle
import time

import numpy as np



def disk_cache(name, sig, compute, force=None):
    """Charge ``{name}.{sig}.pkl`` si présent, sinon (re)calcule et sauve.

    Parameters
    ----------
    name : str
        Préfixe logique du résultat (ex. 'base', 'period_data').
    sig : str
        Signature ; toute variante du résultat doit changer la signature.
    compute : callable
        Fonction sans argument renvoyant l'objet à mettre en cache.
    force : bool or None
        Force le recalcul ; ``None`` -> valeur de ``config.FORCE_RECOMPUTE``.

    Returns
    -------
    object
        L'objet chargé du cache ou fraîchement calculé.
    """
    force = FORCE_RECOMPUTE if force is None else force
    path = os.path.join(CACHE_DIR, f"{name}.{sig}.pkl")
    if not force and os.path.exists(path):
        print(f"[cache] OK  {name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    # une seule version par 'name' : on purge les signatures périmées
    for old in glob.glob(os.path.join(CACHE_DIR, f"{name}.*.pkl")):
        os.remove(old)
    t0 = time.time()
    obj = compute()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[cache] ..  {name} calculé en {time.time() - t0:.1f}s")
    return obj


def sig_of(*parts):
    """Signature md5 courte (10 hex), sensible au CONTENU des arguments.

    Les ``np.ndarray`` sont hachés sur leurs octets et leur forme ; tout autre
    objet via ``repr``. Deux appels avec des contenus identiques donnent la même
    signature, ce qui permet de réutiliser un cache existant à l'identique.

    Parameters
    ----------
    *parts
        Valeurs scalaires, chaînes, tuples ou tableaux numpy.

    Returns
    -------
    str
        10 premiers caractères hexadécimaux du md5.
    """
    h = hashlib.md5()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:10]
