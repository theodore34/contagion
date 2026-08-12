"""Null model 'donnees random, memes matrices' : block-bootstrap des rendements,
re-detection + fit SIS avec les VRAIES matrices A, comptage des courbes R2>seuil.

Le block-bootstrap conserve la correlation contemporaine (donc ~autant de courbes
detectees) mais casse l'ordre temporel -> teste si la dynamique reelle compte.
iid / phase-rand detruisent la correlation (0 courbe), confirmes a part.

Sortie : cache/null_random_blockboot.pkl (real_counts, null_counts par methode).
"""
import sys, pickle, numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.stats import linregress
from joblib import Parallel, delayed

sys.path.insert(0, '.')
import config, data as datamod, periods, sis

_ctx = datamod.build_context()
N, MASK = _ctx.N, _ctx.mask_off
SIS_MODELS = list(_ctx.A_sis.keys())
R2 = config.R2_SEUIL
A_SIS = _ctx.A_sis
REALVALS = _ctx.data.values
IDX, COLS = _ctx.data.index, _ctx.data.columns


def _light_base(vals):
    """jours, mc journalier, E_i journalier depuis un tableau (T, N) aligne sur IDX."""
    df = pd.DataFrame(vals, index=IDX, columns=COLS)
    day = np.array([t.date() for t in IDX])
    days = np.array(sorted(set(day)))
    mc = np.full(len(days), np.nan); E = np.full((N, len(days)), np.nan)
    for k, dd in enumerate(days):
        chunk = vals[day == dd]
        if len(chunk) < 3:
            continue
        C = np.abs(np.nan_to_num(np.corrcoef(chunk.T), nan=0.0)); np.fill_diagonal(C, 0)
        mc[k] = C[MASK].mean(); E[:, k] = C.sum(1) / (N - 1)
    return df, days, mc, E


def _count(vals):
    """Nb de courbes R2>seuil par methode (vraies matrices A, integrate_xscan)."""
    df, days, mc, E = _light_base(vals)
    cm = periods._crisis_map_from(E)
    pk = periods.detect_peaks(mc, float(np.nanmean(mc)))
    pr = periods.peak_rises(pk['peak_intervals'], pk['mc_s'])
    fc = SimpleNamespace(data=df, all_days=days, N=N, crisis_map=cm, E_daily=E, A_sis=A_SIS)
    cnt = {m: 0 for m in SIS_MODELS}; nc = 0
    for (pa, pb) in pr:
        cache = sis.cache_signed(pa, pb, fc)
        for gi in cache:
            E_i = cache[gi]['E_i']; ok = ~np.isnan(E_i)
            if ok.sum() < 3:
                continue
            nc += 1
            for m in SIS_MODELS:
                xt = sis.integrate_xscan(cache[gi], gi, A_SIS[m])
                if linregress(E_i[ok], xt[ok]).rvalue ** 2 > R2:
                    cnt[m] += 1
    return dict(n_curves=nc, n_peaks=len(pr), counts=cnt)


def _block_boot(seed, block=20):
    rng = np.random.default_rng(seed)
    T = REALVALS.shape[0]; nb = T // block
    starts = rng.integers(0, T - block + 1, size=nb)
    out = np.vstack([REALVALS[s:s + block] for s in starts])[:T]
    if out.shape[0] < T:                      # complete a T lignes
        out = np.vstack([out, REALVALS[:T - out.shape[0]]])
    return out


def _one_null(seed):
    return _count(_block_boot(seed))


if __name__ == '__main__':
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    real = _count(REALVALS)
    print('REAL :', real, flush=True)
    res = Parallel(n_jobs=min(NB, config.N_JOBS))(delayed(_one_null)(s) for s in range(NB))
    null = {m: np.array([r['counts'][m] for r in res], float) for m in SIS_MODELS}
    print(f'\n{NB} block-bootstraps (courbes ~{np.mean([r["n_curves"] for r in res]):.0f}/realisation)')
    print(f"{'methode':10s} {'reel':>5s} {'null_moy':>8s} {'null_sd':>7s} {'z':>6s} {'p':>8s}")
    summary = {'real': real['counts'], 'null': {}}
    for m in SIS_MODELS:
        mu, sd = null[m].mean(), null[m].std()
        z = (real['counts'][m] - mu) / (sd + 1e-12)
        p = (1 + np.sum(null[m] >= real['counts'][m])) / (NB + 1)
        summary['null'][m] = null[m]
        print(f'{m:10s} {real["counts"][m]:5d} {mu:8.1f} {sd:7.2f} {z:6.1f} {p:8.4f}')
    pickle.dump(summary, open('../cache/null_random_blockboot.pkl', 'wb'))
    print('\n-> cache/null_random_blockboot.pkl')
