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

from config import CACHE_DIR, FORCE_RECOMPUTE


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
