"""Paths and hyperparameters — the single place to tune the whole pipeline.

Every other module imports its constants from here. Changing a value here
automatically invalidates the affected caches (cache signatures include these
parameters), so it is enough to re-run the relevant script.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent      # repository root
CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data"                            # price CSVs (stock_filled.csv ...)
CACHE_DIR = ROOT / "cache"                          # pickles of heavy results
FIG_DIR = ROOT / "Fig"                              # generated figures
RESULTS_DIR = ROOT / "results"                      # CSV / final-figure exports
for _d in (CACHE_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# Recompute everything, ignoring the cache, when True (set True occasionally).
FORCE_RECOMPUTE = False

# Cap on joblib workers for the parallel sweeps: bounds the RAM
# (each worker duplicates the caches; too many workers => memory blow-up).
N_JOBS = 2

# ── Data ──────────────────────────────────────────────────────────────────────
ASSETS = ["stock"]          # data/{a}_filled.csv files merged on the date
LOG_RETURNS = True
SORT_BY_SECTOR = True

# ── Crisis-window detection (rising correlation / E_i) ────────────────────────
SMOOTH = 14                 # smoothing width (moving average) before detection
MIN_LEN = 12                # minimum length of a rising window (days)
MERGE_GAP = 3               # merge two windows separated by <= this many declining days
REFINE = SMOOTH // 2        # fine edge adjustment (±REFINE) to maximise amplitude
CRISIS_FACTOR = 0.6         # GLOBAL window kept if <|C|>_window > CRISIS_FACTOR x mean
ASSET_FACTOR = 0.6          # asset "in crisis" if E_i_window > ASSET_FACTOR x its mean

# ── Per-asset strong-rise sub-window (strong_rise) ────────────────────────────
SR_SMOOTH = 2               # smoothing of the sub-window
SR_MIN_LEN = 10             # minimum length of the sub-window (days)
SR_MIN_GAIN = 0.5           # minimum gain (rise) required on the sub-window

# ── Contagion network ─────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.85       # quantile of |C| for the 'Corr thr' matrix
SIS_MODELS = ["PMFG", "VAR(1)", "VAR(13)", "Corr thr"]

# ── SIS model ─────────────────────────────────────────────────────────────────
B_FIT, R_FIT = 1.0, 1.0     # recovery rate B and infection rate R
T_LONG = 50.0               # integration horizon to find the equilibrium x*
TOL_EQ = 0.01               # tolerance on x* for the convergence time T_conv

# ── Curve / period selection ──────────────────────────────────────────────────
R2_SEUIL = 0.6              # per-fit R² threshold for retained curves / scatter
N_MIN = 20                  # minimum number of in-crisis assets to keep a period
R2_PERIOD = 0.2             # minimum mean R² (best method) to keep a period

# ── Crisis detection from PEAKS of <|C_ij|> ───────────────────────────────────
PK_PROM_FACTOR = 0.30       # minimum prominence = factor x std of <|C|>
PK_DISTANCE = MIN_LEN       # minimum distance (days) between two peaks
PK_RELHEIGHT = 0.6          # fraction of the prominence -> peak width

# ── Spectral diagnostic λ_max/⟨λ⟩ (as in article.ipynb) ───────────────────────
WINDOW_SPEC = 13 * 21       # ~1 month of 30-min bars
STEP_SPEC = 13              # 1 point / day
SPEC_SMOOTH = 15            # smoothing before detecting λ_max/⟨λ⟩ peaks
SPEC_PROM_FACTOR = 0.30     # prominence of λ_max/⟨λ⟩ peaks (x std)

# ── Experiments ───────────────────────────────────────────────────────────────
import numpy as _np
# 'Corr thr' q-threshold sweep: coarse step (0.05) then fine (0.01) in the jump zone
Q_SWEEP = _np.round(_np.concatenate([_np.arange(0.50, 0.80, 0.05),
                                     _np.arange(0.80, 0.96, 0.01)]), 2)
Q_PANEL = _np.round(_np.arange(0.50, 0.96, 0.05), 2)   # coarse grid for the panel
LAG_SCAN = list(range(1, 30))                          # VAR lags swept for the SIS fit
# SIS-rate sweep: B (recovery) and R (infection), same grid for both
BR_GRID = _np.round(_np.arange(0.80, 1.21, 0.20), 2)   # [0.8, 1.0, 1.2] (reduced: RAM)


# ═══════════════════════════ Disk cache (was cache.py) ═══════════════════════════

"""Minimal disk cache: a heavy computation is done only once.

`disk_cache(name, sig, compute)` loads ``cache/{name}.{sig}.pkl`` if it exists,
otherwise runs ``compute()``, saves the result and returns it. The signature
``sig`` must change as soon as the result would change: use :func:`sig_of` to
build a CONTENT-sensitive signature (np.ndarray included).

Stable naming scheme: pickles already produced are reused as-is across runs.
"""
import glob
import hashlib
import os
import pickle
import time

import numpy as np



def disk_cache(name, sig, compute, force=None):
    """Load ``{name}.{sig}.pkl`` if present, otherwise (re)compute and save.

    Parameters
    ----------
    name : str
        Logical prefix of the result (e.g. 'base', 'period_data').
    sig : str
        Signature; any variation of the result must change the signature.
    compute : callable
        Zero-argument function returning the object to cache.
    force : bool or None
        Force recomputation; ``None`` -> value of ``config.FORCE_RECOMPUTE``.

    Returns
    -------
    object
        The object loaded from cache or freshly computed.
    """
    force = FORCE_RECOMPUTE if force is None else force
    path = os.path.join(CACHE_DIR, f"{name}.{sig}.pkl")
    if not force and os.path.exists(path):
        print(f"[cache] OK  {name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    # one version per 'name': purge stale signatures
    for old in glob.glob(os.path.join(CACHE_DIR, f"{name}.*.pkl")):
        os.remove(old)
    t0 = time.time()
    obj = compute()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[cache] ..  {name} calculé en {time.time() - t0:.1f}s")
    return obj


def sig_of(*parts):
    """Short md5 signature (10 hex), sensitive to the CONTENT of the arguments.

    ``np.ndarray`` are hashed on their bytes and shape; any other object via
    ``repr``. Two calls with identical contents yield the same signature, which
    lets an existing cache be reused unchanged.

    Parameters
    ----------
    *parts
        Scalar values, strings, tuples or numpy arrays.

    Returns
    -------
    str
        First 10 hexadecimal characters of the md5.
    """
    h = hashlib.md5()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:10]
