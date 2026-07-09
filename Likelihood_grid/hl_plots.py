#!/usr/bin/env python3
"""
hl_plots.py  —  diagnostic and posterior plots for the HL HM1×HM1 likelihood.

All functions save to output_dir and return None.

Diagnostic (call BEFORE running the likelihood to verify inputs):
    plot_likelihood_inputs(data, config, output_dir)
        Figure 1: spectra — sims, mean±σ/√N, model r=0, noise, fg
        Figure 2: normalised residuals per pair, χ²/dof colour-coded
        Figure 3: 22×22 pair pull-matrix heatmap

    plot_pair_pull_matrix(data, config, output_dir)
        Standalone 22×22 heatmap (also called inside plot_likelihood_inputs).

Results (call AFTER run_hl_chi2):
    plot_posterior_single(ip_main, pps, chi2, r_grid, fine, ...)
        Posterior with per-sim overlays and statistics text box.

Legacy quick-look:
    plot_model_vs_data(cldata, fiducial, ell, ch_names, n_hm1, output_dir)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from hl_stats import idx2chs, upscale_posterior


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shorten(ch_name):
    """'LFT_040' → 'L040',  'MFT_100' → 'M100',  'HFT_235' → 'H235'."""
    p = ch_name.split('_')
    return p[0][0] + p[1] if len(p) == 2 else ch_name


def _compute_pulls(cldata, fiducial):
    """
    Normalised residuals: (mean − fiducial) / (σ / √N).

    Parameters
    ----------
    cldata   : (N_data, N_pairs, N_ell)
    fiducial : (N_pairs, N_ell)

    Returns
    -------
    pull     : (N_pairs, N_ell)
    max_pull : (N_pairs,)  — max |pull| over ell bins per pair
    """
    n     = cldata.shape[0]
    mean  = cldata.mean(axis=0)
    sigma = cldata.std(axis=0) / np.sqrt(n)
    sigma = np.where(sigma < 1e-30, 1e-30, sigma)
    pull  = (mean - fiducial) / sigma
    return pull, np.abs(pull).max(axis=1)


def _posterior_stats(ip, fine):
    """
    Basic posterior statistics using numpy only (no scipy).

    Returns dict: peak, mean, sigma, lo68, hi68, ul95 (equal-tail).
    """
    norm = float(np.trapezoid(ip, fine)) or 1.0
    peak = float(fine[np.argmax(ip)])
    mean = float(np.trapezoid(fine * ip, fine) / norm)
    var  = float(np.trapezoid((fine - mean) ** 2 * ip, fine) / norm)
    sig  = float(np.sqrt(max(var, 0.0)))
    cum  = np.cumsum(ip); cum /= cum[-1]
    lo68 = float(fine[np.searchsorted(cum, 0.1585)])
    hi68 = float(fine[np.searchsorted(cum, 0.8415)])
    ul95 = float(fine[np.searchsorted(cum, 0.95)])
    return dict(peak=peak, mean=mean, sigma=sig, lo68=lo68, hi68=hi68, ul95=ul95)


# ---------------------------------------------------------------------------
# plot_pair_pull_matrix — 22×22 heatmap of max |pull| per pair
# ---------------------------------------------------------------------------

def plot_pair_pull_matrix(data, config, output_dir):
    """
    22×22 heatmap of max |pull| = max_ℓ |(mean−fid)/(σ/√N)| for each pair.

    Rows and columns follow the same channel ordering as hm1_channels.
    Diagonal cells = AUTO spectra (thick border).
    Vertical/horizontal dividers mark LFT | MFT | HFT telescope boundaries.
    Colormap: white = 0,  yellow = 1.5,  red ≥ 3.

    A channel whose entire row (or column) is red signals a problem with
    the noise/fg model for that detector.
    """
    cldata   = data["cldata"]
    fiducial = data["fiducial"]
    n_hm1    = data["N_fields"]
    n_pairs  = n_hm1 * (n_hm1 + 1) // 2
    ell_min  = int(data["ell"][0])
    ch_names = list(config["hm1_channels"].keys())
    labels   = [_shorten(n) for n in ch_names]

    _, max_pull = _compute_pulls(cldata, fiducial)

    mat = np.zeros((n_hm1, n_hm1))
    for k in range(n_pairs):
        i, j = idx2chs(k, n_hm1)
        mat[i, j] = max_pull[k]
        mat[j, i] = max_pull[k]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=120, constrained_layout=True)
    im = ax.imshow(mat, vmin=0, vmax=3, cmap=plt.cm.RdYlGn_r,
                   aspect='equal', origin='upper')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('max |pull| over ℓ bins', fontsize=9)
    cbar.ax.axhline(2.0, color='black', lw=1.5, ls='--')
    cbar.ax.text(2.5, 2.05, '2σ', va='bottom', fontsize=7)

    # thick border on diagonal (AUTO pairs)
    for i in range(n_hm1):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='black', lw=2.0))

    # telescope dividers: LFT 0-11, MFT 12-17, HFT 18-21
    for b in [12, 18]:
        ax.axhline(b - 0.5, color='white', lw=2.5)
        ax.axvline(b - 0.5, color='white', lw=2.5)

    # telescope labels on right margin
    scope_info = [("LFT", 0, 12), ("MFT", 12, 18), ("HFT", 18, n_hm1)]
    for sname, s0, s1 in scope_info:
        mid = (s0 + s1 - 1) / 2
        ax.text(n_hm1 + 0.3, mid, sname, va='center', fontsize=7,
                color='dimgray', transform=ax.transData)

    ax.set_xticks(range(n_hm1)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_hm1)); ax.set_yticklabels(labels, fontsize=6)

    n_warn = int((max_pull > 2.0).sum())
    ax.set_title(
        f'Pair pull matrix — max |pull| over ℓ  [ellmin={ell_min}]\n'
        f'{n_warn}/{n_pairs} pairs with max|pull| > 2'
        f'  (diagonal = AUTO, colour: 0=white, 3=red)',
        fontsize=10,
    )

    out = f'{output_dir}/pair_pull_matrix_ellmin{ell_min}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# plot_likelihood_inputs — three-figure diagnostic
# ---------------------------------------------------------------------------

def plot_likelihood_inputs(data, config, output_dir, n_plot=12):
    """
    Three diagnostic figures verifying everything that enters the likelihood.

    Figure 1 — spectra (n_plot panels, balanced AUTO+cross selection):
        Gray lines      : DLcross test sims (first 50)
        Shaded band     : mean ± σ/√N
        Blue solid      : mean DLcross
        Red dashed      : fiducial r=0 = noise_mean + fg
        Orange dotted   : noise_mean (detector noise contribution)
        Green dotted    : foreground model

    Figure 2 — normalised residuals:
        (mean − fiducial) / (σ/√N) per pair and ℓ bin
        ±1σ light-grey shaded band
        ±2σ red dashed horizontal lines
        Panel title colour: green = χ²/dof < 1.3  |  orange < 2.5  |  red ≥ 2.5

    Figure 3 — 22×22 pair pull matrix heatmap.
    """
    pairs_mode = data["pairs_mode"]
    ell        = data["ell"]
    ell_min    = int(ell[0])
    cldata     = data["cldata"]        # (N_data, 253, n_ell)
    fiducial   = data["fiducial"]      # (253, n_ell)
    noise_mean = data["noise_mean"]
    fg         = data["fg"]
    n_hm1      = data["N_fields"]
    ch_names   = list(config["hm1_channels"].keys())
    n_pairs    = n_hm1 * (n_hm1 + 1) // 2

    # ---- balanced pair selection: first n_plot//2 AUTO + n_plot//2 cross ----
    auto_idxs  = list(data["auto_pair_indices"])
    cross_idxs = list(data["cross_pair_indices"])
    n_auto     = min(len(auto_idxs),  n_plot // 2)
    n_cross    = min(len(cross_idxs), n_plot - n_auto)
    plot_pairs = auto_idxs[:n_auto] + cross_idxs[:n_cross]
    n_plot_eff = len(plot_pairs)

    pull_all, max_pull_all = _compute_pulls(cldata, fiducial)
    n_warn_global = int((max_pull_all > 2.0).sum())

    ncols = 4
    nrows = max(1, int(np.ceil(n_plot_eff / ncols)))

    # ---------------------------------------------------------------- #
    # Figure 1: spectra                                                 #
    # ---------------------------------------------------------------- #
    fig1, axes1 = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 3.5 * nrows), dpi=100,
                               constrained_layout=True, squeeze=False)
    axes1 = axes1.flatten()

    for k, p_idx in enumerate(plot_pairs):
        ax      = axes1[k]
        i, j    = idx2chs(int(p_idx), n_hm1)
        lbl     = f"{_shorten(ch_names[i])}×{_shorten(ch_names[j])}"
        is_auto = (i == j)

        sims = cldata[:, p_idx, :]
        mean = sims.mean(axis=0)
        sig  = sims.std(axis=0) / np.sqrt(sims.shape[0])
        chi2_panel = float(np.sum(pull_all[p_idx] ** 2) / len(ell))

        for s in range(min(sims.shape[0], 50)):
            ax.plot(ell, sims[s], color='gray', lw=0.4, alpha=0.2, zorder=1)

        ax.fill_between(ell, mean - sig, mean + sig,
                        color='steelblue', alpha=0.25, zorder=2, label='mean ± σ/√N')
        ax.plot(ell, mean,              color='steelblue',   lw=1.5,              zorder=3)
        ax.plot(ell, fiducial[p_idx],   color='red',         lw=1.8, ls='--',     zorder=4,
                label='fid (n+fg)')
        ax.plot(ell, noise_mean[p_idx], color='darkorange',  lw=1.1, ls=':',      zorder=4,
                label='noise_mean')
        ax.plot(ell, fg[p_idx],         color='forestgreen', lw=1.1, ls=':',      zorder=4,
                label='fg')

        title_col = ('green' if chi2_panel < 1.3
                     else 'darkorange' if chi2_panel < 2.5
                     else 'red')
        ax.set_title(
            f'{"★ " if is_auto else ""}{lbl}  χ²/dof={chi2_panel:.2f}',
            fontsize=7, color=title_col,
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r'$\ell$', fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel(r'$D_\ell^{BB}$ [$\mu K^2$]', fontsize=7)
        if k == 0:
            ax.legend(fontsize=5.5, frameon=False, ncol=2, loc='upper right')

    for k in range(n_plot_eff, len(axes1)):
        axes1[k].set_visible(False)

    fig1.suptitle(
        f'HL inputs — spectra  [{pairs_mode} pairs, ellmin={ell_min}]   '
        f'N_data={data["N_data"]}   '
        f'{n_warn_global}/{n_pairs} pairs with max|pull|>2\n'
        f'★ = AUTO  |  title green: χ²/dof<1.3  orange: <2.5  red: ≥2.5',
        fontsize=9,
    )
    out1 = f'{output_dir}/likelihood_inputs_spectra_{pairs_mode}_ellmin{ell_min}.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # ---------------------------------------------------------------- #
    # Figure 2: normalised residuals                                    #
    # ---------------------------------------------------------------- #
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 3.0 * nrows), dpi=100,
                               constrained_layout=True, squeeze=False)
    axes2 = axes2.flatten()

    for k, p_idx in enumerate(plot_pairs):
        ax      = axes2[k]
        i, j    = idx2chs(int(p_idx), n_hm1)
        lbl     = f"{_shorten(ch_names[i])}×{_shorten(ch_names[j])}"
        is_auto = (i == j)

        pull     = pull_all[p_idx]
        chi2_red = float(np.sum(pull ** 2) / len(ell))
        title_col = ('green' if chi2_red < 1.3
                     else 'darkorange' if chi2_red < 2.5
                     else 'red')

        ax.axhspan(-1.0, 1.0, color='lightgray', alpha=0.4, zorder=1)
        ax.axhline(0.0,  color='gray', ls=':',  lw=0.8, zorder=2)
        ax.axhline(+2.0, color='red',  ls='--', lw=0.9, zorder=2)
        ax.axhline(-2.0, color='red',  ls='--', lw=0.9, zorder=2)

        col = 'firebrick' if is_auto else 'steelblue'
        ax.plot(ell, pull, 'o-', color=col, ms=3.5, lw=1.2, zorder=3)

        tag = "★ AUTO" if is_auto else "cross"
        ax.set_title(
            f'{lbl} [{tag}]  χ²/dof = {chi2_red:.2f}',
            fontsize=7, color=title_col,
        )
        ax.tick_params(labelsize=7)
        ax.set_ylabel(
            r'$(\bar{D}_\ell - D^{\rm fid}_\ell)\,/\,(\sigma/\sqrt{N})$',
            fontsize=6,
        )
        if k >= n_plot_eff - ncols:
            ax.set_xlabel(r'$\ell$', fontsize=8)

    for k in range(n_plot_eff, len(axes2)):
        axes2[k].set_visible(False)

    n_red = sum(
        1 for p in plot_pairs
        if float(np.sum(pull_all[p] ** 2) / len(ell)) >= 2.5
    )
    fig2.suptitle(
        f'HL inputs — normalised residuals  [{pairs_mode} pairs, ellmin={ell_min}]\n'
        f'Grey band = ±1σ  |  Red dashes = ±2σ  |  '
        f'{n_red}/{n_plot_eff} shown panels with χ²/dof ≥ 2.5',
        fontsize=9,
    )
    out2 = f'{output_dir}/likelihood_inputs_residuals_{pairs_mode}_ellmin{ell_min}.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ---------------------------------------------------------------- #
    # Figure 3: 22×22 pair pull matrix                                 #
    # ---------------------------------------------------------------- #
    plot_pair_pull_matrix(data, config, output_dir)


# ---------------------------------------------------------------------------
# plot_posterior_single — posterior with statistics text box
# ---------------------------------------------------------------------------

def plot_posterior_single(ip_main, pps, chi2, r_grid, fine, n_pairs, n_data,
                          ell_bins_keep, output_dir, ell_min, label="HL"):
    """
    Single posterior panel with per-sim overlays and a statistics text box.

    Statistics shown: peak r, mean r, σ, 68% equal-tail CI, 95% upper limit.

    Parameters
    ----------
    ip_main       : (N_fine,)        mean-chi2 posterior on fine grid
    pps           : (N_r, N_data)    per-sim posteriors (coarse r_grid)
    chi2          : (N_r, N_data)    chi2 array
    r_grid        : (N_r,)
    fine          : (N_fine,)
    n_pairs       : int
    n_data        : int
    ell_bins_keep : list
    output_dir    : str
    ell_min       : int
    label         : str              run label (pairs_mode + offset_type)
    """
    stats = _posterior_stats(ip_main, fine)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    # per-sim posteriors (thin grey)
    for i in range(chi2.shape[1]):
        p = upscale_posterior(pps[:, i], r_grid, fine)
        ax.plot(fine, p, color='#9AA5B1', lw=0.7, alpha=0.30,
                label='single sim' if i == 0 else None)

    # mean-chi2 posterior
    ax.plot(fine, ip_main, color='#6b1a6b', lw=2.2, label=f'HL — {label} (mean χ²)')

    # r_true = 0 reference
    ax.axvline(0.0, color='gray', ls=':', lw=1.0, alpha=0.7, label=r'$r_{\rm true}=0$')

    # 68% CI shading
    ax.axvspan(stats["lo68"], stats["hi68"], alpha=0.12, color='#6b1a6b',
               label=f'68% CI')

    # statistics text box (top-right)
    peak_at_zero = stats["peak"] < (fine[1] - fine[0]) * 2
    if peak_at_zero:
        stat_lines = (
            f"Peak  = {stats['peak']:.5f}\n"
            f"Mean  = {stats['mean']:.5f}\n"
            f"σ     = {stats['sigma']:.5f}\n"
            f"68% CI: [{stats['lo68']:.5f}, {stats['hi68']:.5f}]\n"
            f"95% UL = {stats['ul95']:.5f}"
        )
    else:
        stat_lines = (
            f"Peak  = {stats['peak']:.5f}\n"
            f"Mean  = {stats['mean']:.5f}\n"
            f"σ     = {stats['sigma']:.5f}\n"
            f"68% CI: [{stats['lo68']:.5f}, {stats['hi68']:.5f}]\n"
            f"95% CI: [{stats['lo68']:.5f}, {stats['ul95']:.5f}]"
        )
    ax.text(0.97, 0.95, stat_lines, transform=ax.transAxes,
            fontsize=8, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#6b1a6b', alpha=0.85))

    ax.set_xlabel('$r$', fontsize=12)
    ax.set_ylabel('Posterior (normalised)', fontsize=12)
    ax.set_title(
        f'HL {label} — {n_pairs} HM1×HM1 pairs  N_data={n_data}\n'
        f'ell bins {ell_bins_keep}',
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = (f'{output_dir}/hl_posterior_'
           f'{label.replace(" ", "_").replace("/", "-")}_ellmin{ell_min}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# plot_model_vs_data — quick 3×4 spectral overview
# ---------------------------------------------------------------------------

def plot_model_vs_data(cldata_hm1, clfidu_hm1, ell, ch_names, n_hm1, output_dir,
                       n_plot=12):
    """
    Quick 3×4 grid: r=0 model vs data sims for the first n_plot HM1 pairs.

    Parameters
    ----------
    cldata_hm1 : (N_data, N_pairs, N_ell)
    clfidu_hm1 : (N_pairs, N_ell)
    ell        : (N_ell,)
    ch_names   : list of str  (length n_hm1)
    n_hm1      : int
    output_dir : str
    n_plot     : int  (default 12)
    """
    n_plot = min(n_plot, cldata_hm1.shape[1])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=150,
                             constrained_layout=True)
    axes = axes.flatten()

    for k in range(n_plot):
        ax    = axes[k]
        i, j  = idx2chs(k, n_hm1)
        label = (f"{ch_names[i]} AUTO"
                 if i == j else
                 f"{_shorten(ch_names[i])}×{_shorten(ch_names[j])}")
        for n in range(cldata_hm1.shape[0]):
            ax.plot(ell, cldata_hm1[n, k, :],
                    color='#9AA5B1', lw=0.5, alpha=0.25)
        ax.plot(ell, cldata_hm1[:, k, :].mean(axis=0),
                color='steelblue', lw=1.5, label='mean data')
        ax.plot(ell, clfidu_hm1[k, :],
                color='red', lw=2.0, ls='--', label='model r=0')
        ax.set_title(label, fontsize=7)
        ax.set_xlabel(r'$\ell$', fontsize=8)
        if k % 4 == 0:
            ax.set_ylabel(r'$D_\ell^{BB}$', fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Model r=0 vs data — first {n_plot} HM1 pairs', fontsize=11)
    ell_min = int(ell[0])
    out = f'{output_dir}/model_vs_data_hm1_ellmin{ell_min}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")
