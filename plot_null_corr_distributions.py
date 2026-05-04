"""Trace la distribution des coefficients de corrélation off-diagonaux
pour la matrice réelle et les 3 modèles nuls."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import load_data


def gen_iid_gaussian(real, rng):
    sigmas = real.std(axis=0).values
    arr = rng.standard_normal(real.shape) * sigmas
    return pd.DataFrame(arr, index=real.index, columns=real.columns)


def gen_phase_randomized(real, rng):
    arr = real.values
    T_, N_ = arr.shape
    out = np.empty_like(arr)
    for j in range(N_):
        Fx = np.fft.rfft(arr[:, j])
        amps = np.abs(Fx)
        phases = rng.uniform(0, 2 * np.pi, size=Fx.shape)
        phases[0] = 0
        if T_ % 2 == 0:
            phases[-1] = 0
        out[:, j] = np.fft.irfft(amps * np.exp(1j * phases), n=T_)
    return pd.DataFrame(out, index=real.index, columns=real.columns)


def gen_block_bootstrap(real, rng, block=20):
    arr = real.values
    T_ = arr.shape[0]
    n_blocks = T_ // block
    starts = rng.integers(0, T_ - block + 1, size=n_blocks)
    out = np.vstack([arr[s:s + block] for s in starts])[:T_]
    return pd.DataFrame(out, index=real.index[:out.shape[0]], columns=real.columns)


def offdiag(C):
    N = C.shape[0]
    return C[np.triu_indices(N, 1)]


def main():
    rng = np.random.default_rng(42)
    real = load_data(['stock'], log_returns=True, sort_by_sector=True)
    N = real.shape[1]

    datasets = {
        'real':        real,
        'iid':         gen_iid_gaussian(real, rng),
        'phase-rand':  gen_phase_randomized(real, rng),
        'block-boot':  gen_block_bootstrap(real, rng, block=20),
    }

    colors = {
        'real':       '#000000',
        'iid':        '#2ca02c',
        'phase-rand': '#1f77b4',
        'block-boot': '#d62728',
    }

    corrs = {name: np.corrcoef(d.values.T) for name, d in datasets.items()}
    offs = {name: offdiag(C) for name, C in corrs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- panneau 1 : densité (signed) ---
    ax = axes[0]
    bins = np.linspace(-0.4, 0.9, 80)
    for name, x in offs.items():
        ax.hist(x, bins=bins, density=True, histtype='step', linewidth=2,
                color=colors[name], label=f'{name}  (mean |ρ|={np.abs(x).mean():.3f})')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(r'$\rho_{ij}$ (off-diagonal)')
    ax.set_ylabel('density')
    ax.set_title('Distribution des corrélations contemporaines')
    ax.legend(frameon=False, fontsize=9)

    # --- panneau 2 : |rho| en log-y pour bien voir les queues ---
    ax = axes[1]
    bins = np.linspace(0, 0.9, 60)
    for name, x in offs.items():
        ax.hist(np.abs(x), bins=bins, density=True, histtype='step',
                linewidth=2, color=colors[name], label=name)
    ax.set_yscale('log')
    ax.set_xlabel(r'$|\rho_{ij}|$')
    ax.set_ylabel('density (log)')
    ax.set_title(r'Queue des $|\rho|$ — pourquoi block-boot reproduit la structure')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f'Stocks N={N}, T={len(real)}  ·  off-diagonal corrélation contemporaine',
                 fontsize=11)
    fig.tight_layout()

    out_path = 'results/null_corr_distributions.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out_path}')

    print('\nRécapitulatif (mean |ρ| off-diagonal) :')
    for name, x in offs.items():
        print(f'  {name:12s}  mean |ρ| = {np.abs(x).mean():.3f}   '
              f'std ρ = {x.std():.3f}   max |ρ| = {np.abs(x).max():.3f}')


if __name__ == '__main__':
    main()
