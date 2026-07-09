#!/usr/bin/env python3
"""
hl_stats.py
===========
Self-contained statistics module for the HL HM1×HM1 pipeline.

Sections
--------
  1. PAIR INDEXING          — map (ch_i, ch_j) ↔ flat pair index k
  2. HL SCALAR TRANSFORM    — ghl(x) used inside the chi2 kernel
  3. POSTERIOR UTILITIES    — interpolation, moments, HDI
  4. OFFSET COMPUTATION     — lollipop and Monte Carlo offsets
  5. HL CHI2 ENGINE         — CHARM-Like (Hamimeche & Lewis 2008 eq. 16)
  6. POSTERIOR SUMMARY      — posterior_from_chi2, print_stats
  7. MAIN ENTRY POINT       — build_config, run_hl_chi2
"""

import os as _os
import numpy as np
from scipy.interpolate import UnivariateSpline

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): yield from it


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAIR INDEXING  (was: common_functions_HL.py)
#    Upper-triangular bijection between channel pairs (i,j) and a flat index k.
#    k = chs2idx(i, j, N) = i*(2N - i + 1)//2 + (j - i)
# ═══════════════════════════════════════════════════════════════════════════════

def chs2idx(ch1, ch2, N_chs):
    """Pair (ch1, ch2) with ch1 <= ch2 → flat upper-triangle index."""
    return ch1 * (2 * N_chs - ch1 + 1) // 2 + (ch2 - ch1)


def idx2chs(idx, N_chs):
    """Flat upper-triangle index → pair (i, j) with i <= j."""
    total = N_chs * (N_chs + 1) // 2
    if idx < 0 or idx >= total:
        raise ValueError(f"Index {idx} out of bounds for {N_chs} channels.")
    i = 0
    while (i * (2 * N_chs - i + 1)) // 2 <= idx:
        i += 1
    i -= 1
    j = i + (idx - i * (2 * N_chs - i + 1) // 2)
    return i, j


def vec2mat(vect, N_fields, *, all=True, cross=True):
    """Upper-triangle vector (N_pairs,) or (N_sims, N_pairs) → (N, N) or (N_sims, N, N)."""
    cross_idxs = np.array([chs2idx(ch1, ch2, N_fields)
                            for ch1 in range(N_fields)
                            for ch2 in range(ch1 + 1, N_fields)])
    auto_idxs  = np.array([chs2idx(ch, ch, N_fields) for ch in range(N_fields)])
    if len(vect.shape) == 2:
        mat = np.zeros((vect.shape[0], N_fields, N_fields))
    else:
        mat = np.zeros((1, N_fields, N_fields))
        vect = vect[None, :]
    if all:
        for i in auto_idxs:
            ch1, ch2 = idx2chs(i, N_fields)
            mat[:, ch1, ch2] = vect[:, i]
    if cross:
        for i in cross_idxs:
            ch1, ch2 = idx2chs(i, N_fields)
            mat[:, ch1, ch2] = mat[:, ch2, ch1] = vect[:, i]
    return np.squeeze(mat)


def mat2vec(mat, *, all=True, cross=True):
    """(N_sims, N, N) matrix → upper-triangle vector (N_sims, N_pairs)."""
    N_chs = mat.shape[1]
    vec = []
    for idx in range(N_chs * (N_chs + 1) // 2):
        ch1, ch2 = idx2chs(idx, N_chs)
        if cross:
            if all:
                vec.append(mat[:, ch1, ch2])
            elif ch1 != ch2:
                vec.append(mat[:, ch1, ch2])
        elif ch1 == ch2:
            vec.append(mat[:, ch1, ch1])
    return np.array(vec).T


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HL SCALAR TRANSFORM  (was: common_functions_HL.py)
#    ghl(x) is the scalar function applied element-wise inside the chi2 kernel.
#    ghl(x) = sign(x-1) * sqrt( 2*(x - ln(x) - 1) )
# ═══════════════════════════════════════════════════════════════════════════════

def ghl(x):
    """Hamimeche-Lewis scalar: sign(x-1) * sqrt(2*(x - log(x) - 1))."""
    return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POSTERIOR UTILITIES  (was: stats_utils.py)
#    Smoothing, moments, and credible intervals on a 1-D posterior.
# ═══════════════════════════════════════════════════════════════════════════════

def upscale_posterior(posterior, grid, fine_grid):
    """Interpolate posterior onto a finer grid; clips negative values to 0."""
    interp = UnivariateSpline(grid, posterior, s=0)
    fine   = np.maximum(interp(fine_grid), 0.0)
    return fine / fine.max()


def compute_first_moment(posterior, grid):
    """Mean of the posterior distribution."""
    return np.sum(grid * posterior) / np.sum(posterior)


def compute_second_moment_sigma(posterior, grid):
    """Standard deviation of the posterior distribution."""
    mu  = compute_first_moment(posterior, grid)
    mu2 = np.sum(grid**2 * posterior) / np.sum(posterior)
    return np.sqrt(mu2 - mu**2)


def compute_hdi(posterior, grid, cred_mass=0.95):
    """Highest Density Interval at the given credible mass."""
    p          = posterior / posterior.sum()
    sorted_idx = np.argsort(p)[::-1]
    cutoff     = np.argmax(np.cumsum(p[sorted_idx]) >= cred_mass)
    hdi_idx    = sorted_idx[:cutoff + 1]
    return grid[hdi_idx].min(), grid[hdi_idx].max()


# ---------------------------------------------------------------------------
# Offset computation
# ---------------------------------------------------------------------------

def compute_lollipop_like_offset(ell, varcl, clref, fsky=1.0, n_iter=1):
    """
    Lollipop offset (Tristram et al. 2112.07961 / planck-npipe/lollipop).

    Parameters
    ----------
    ell   : (N_ell,)             effective multipoles
    varcl : (N_pairs, N_ell)     empirical variance of Cl
    clref : (N_pairs, N_ell)     reference (fiducial) spectrum
    fsky  : float
    n_iter: int                  number of self-consistency iterations

    Returns
    -------
    offset : (N_pairs, N_ell)
    """
    Nl = np.sqrt(np.abs(varcl - (2.0 / (2.0 * ell + 1) * clref ** 2) / fsky))
    for _ in range(n_iter):
        Nl = np.sqrt(
            np.abs(varcl - 2.0 / (2.0 * ell + 1) / fsky
                   * (clref ** 2 + 2.0 * Nl * clref))
        )
    return Nl * np.sqrt((2.0 * ell + 1) / 2.0)


def compute_lollipop_offset(cldata, clfidu, ell, fsky, n_iter):
    """
    Compute the lollipop offset from the empirical variance of the test data.

    Parameters
    ----------
    cldata : (N_data, N_pairs, N_ell)
    clfidu : (N_pairs, N_ell)
    ell    : (N_ell,)
    fsky   : float
    n_iter : int

    Returns
    -------
    cloff_lol : (N_pairs, N_ell)
    """
    varcl = cldata.var(axis=0)   # (N_pairs, N_ell)
    return compute_lollipop_like_offset(ell, varcl, clfidu, fsky, n_iter)


def compute_mc_offset(cldata_all):
    """
    Monte Carlo offset: for each (pair, ell) entry the 99th percentile of
    |Cl| computed over all simulations where Cl < 0.

    Parameters
    ----------
    cldata_all : (N_sims, N_pairs, N_ell)

    Returns
    -------
    cloff_mc : (N_pairs, N_ell)
    """
    n_pairs = cldata_all.shape[1]
    n_ell   = cldata_all.shape[2]
    cloff_mc = np.zeros((n_pairs, n_ell))
    for k in range(n_pairs):
        for e in range(n_ell):
            spec = cldata_all[:, k, e]
            neg  = np.abs(spec[spec < 0])
            cloff_mc[k, e] = np.quantile(neg, 0.99) if len(neg) > 0 else 0.0
    return cloff_mc


# ---------------------------------------------------------------------------
# HL chi2 engine — private helpers
# ---------------------------------------------------------------------------

_IDX_CACHE: dict = {}


def _build_idx(N):
    """Row/col arrays for upper-triangular indexing of an N×N matrix."""
    Np   = N * (N + 1) // 2
    rows = np.empty(Np, dtype=np.intp)
    cols = np.empty(Np, dtype=np.intp)
    k    = 0
    for i in range(N):
        for j in range(i, N):
            rows[k] = i; cols[k] = j; k += 1
    return rows, cols


def _get_idx(N):
    if N not in _IDX_CACHE:
        _IDX_CACHE[N] = _build_idx(N)
    return _IDX_CACHE[N]


def _v2m(v, N):
    """Upper-triangle vector → symmetric N×N matrix."""
    r, c = _get_idx(N)
    m    = np.zeros((N, N), dtype=v.dtype)
    m[r, c] = v; m[c, r] = v
    return m


def _v2m_batch(vb, N):
    """Batch upper-triangle vectors: (n_sims, N_pairs) → (n_sims, N, N)."""
    r, c = _get_idx(N)
    m    = np.zeros((vb.shape[0], N, N), dtype=vb.dtype)
    m[:, r, c] = vb; m[:, c, r] = vb
    return m


def _precompute_D_LF(cldata, cloff, clfidu, N):
    """
    Pre-compute the offset matrix, D = data + offset, and L_F (square root
    of the fiducial matrix) for each ell bin.

    Returns
    -------
    Off_all : (N_ell, N, N)
    D_all   : (N_ell, N_data, N, N)
    LF_all  : (N_ell, N, N)
    """
    Ne = cldata.shape[2]
    Nd = cldata.shape[0]
    Off_all = np.empty((Ne, N, N))
    D_all   = np.empty((Ne, Nd, N, N))
    LF_all  = np.empty((Ne, N, N))
    for e in range(Ne):
        Off        = _v2m(cloff[:, e], N)
        Off_all[e] = Off
        D_all[e]   = _v2m_batch(cldata[:, :, e], N) + Off
        F0         = _v2m(clfidu[:, e], N) + Off
        w, V       = np.linalg.eigh(F0)
        if np.any(w <= 0):
            print(f"  [WARN] F0 not PSD at ell_idx={e}  min_eig={w.min():.3e}", flush=True)
        LF_all[e]  = (V * np.sqrt(np.maximum(w, 0.0))) @ V.T
    return Off_all, D_all, LF_all


def _precompute_LM(clth, Off_all, N):
    """
    Pre-compute L_M^{-1/2} = (V * 1/sqrt(|w|)) V^T for each (r, ell) bin.

    Returns
    -------
    LM : (N_r, N_ell, N, N)
    """
    Nr = clth.shape[0]
    Ne = clth.shape[2]
    LM = np.empty((Nr, Ne, N, N))
    for r in range(Nr):
        for e in range(Ne):
            M0   = _v2m(clth[r, :, e], N) + Off_all[e]
            w, V = np.linalg.eigh(M0)
            if r == 0 and np.any(w <= 0):
                print(f"  [WARN] M0 not PSD at r=0, ell_idx={e}  min_eig={w.min():.3e}", flush=True)
            LM[r, e] = (V * (1.0 / np.sqrt(np.abs(w) + 1e-60))) @ V.T
    return LM


# ---------------------------------------------------------------------------
# HL chi2
# ---------------------------------------------------------------------------

def compute_hl_chi2_charmlike(clth, cldata, cloff, clfidu, icov, N_fields,
                               sim_batch_size=20, custom_pair_idxs=None):
    """
    HL chi2 following the CHARM-Like approach (Hamimeche & Lewis 2008, eq. 16).

    Parameters
    ----------
    clth             : (N_r, N_pairs, N_ell)    theoretical model (r-dependent)
    cldata           : (N_data, N_pairs, N_ell)  data simulations
    cloff            : (N_pairs, N_ell)          non-negativity offset
    clfidu           : (N_pairs, N_ell)          fiducial r=0 spectrum
    icov             : (N_flat, N_flat)           inverse covariance, pair-major
    N_fields         : int                        number of fields (e.g. 22)
    sim_batch_size   : int                        sims per batch (memory control)
    custom_pair_idxs : array or None             if given, select only these pair
                                                 indices from the X upper-triangle

    Returns
    -------
    chi2 : (N_r, N_data)
    """
    Nr, Nd, Ne = clth.shape[0], cldata.shape[0], cldata.shape[2]
    ri, ci = _get_idx(N_fields)

    print("  Precomputing D, LF...", flush=True)
    Off_all, D_all, LF_all = _precompute_D_LF(cldata, cloff, clfidu, N_fields)
    print("  Precomputing LM...", flush=True)
    LM_all = _precompute_LM(clth, Off_all, N_fields)

    chi2 = np.empty((Nd, Nr))
    nb   = (Nd + sim_batch_size - 1) // sim_batch_size

    for bk in tqdm(range(nb), desc="  [HL chi2] batch", unit="batch", dynamic_ncols=True):
        s0, s1 = bk * sim_batch_size, min((bk + 1) * sim_batch_size, Nd)
        ns     = s1 - s0
        Db     = D_all[:, s0:s1]             # (Ne, ns, N, N)

        tmp = np.einsum("esjl,relk->resjk", Db, LM_all)        # D @ L_M
        P   = np.einsum("reji,resjk->resik", LM_all, tmp); del tmp
        Pb  = P.reshape(-1, N_fields, N_fields); del P
        w, V = np.linalg.eigh(Pb)
        gg  = np.sign(w) * ghl(np.abs(w))
        G   = np.einsum("nij,nj,nkj->nik", V, gg, V); del w, V, Pb
        G   = G.reshape(Nr, Ne, ns, N_fields, N_fields)

        tmp2 = np.einsum("resjl,elk->resjk", G, LF_all)
        X    = np.einsum("eji,resjk->resik", LF_all, tmp2); del G, tmp2
        xp   = X[:, :, :, ri, ci]; del X            # (Nr, Ne, ns, Np)

        if custom_pair_idxs is not None:
            xp = xp[:, :, :, custom_pair_idxs]

        # pair-major flattening: (Nr, Ne, ns, Np) → (ns, Nr, Np*Ne)
        xf   = xp.transpose(2, 0, 3, 1).reshape(ns, Nr, -1); del xp
        tmp2 = xf @ icov
        chi2[s0:s1] = (tmp2 * xf).sum(axis=-1)
        del xf, tmp2

    return chi2.T   # (N_r, N_data)


# ---------------------------------------------------------------------------
# Posterior utilities
# ---------------------------------------------------------------------------

def posterior_from_chi2(chi2_arr):
    """Mean chi2 across sims → normalised posterior on the coarse r grid."""
    mc = chi2_arr.mean(axis=1)
    p  = np.exp(-0.5 * (mc - mc.min()))
    return p / p.max()


def per_sim_posteriors(chi2_arr):
    """Per-simulation posteriors, shape (N_r, N_data)."""
    p = np.exp(-0.5 * (chi2_arr - chi2_arr.min(axis=0)[np.newaxis, :]))
    return p / p.max(axis=0)[np.newaxis, :]


def print_stats(label, chi2_arr, r_grid, r_min, r_max):
    """
    Print posterior statistics (peak, mean, sigma, HDI) for a chi2 array.

    Parameters
    ----------
    label    : str
    chi2_arr : (N_r, N_data)
    r_grid   : (N_r,)
    r_min    : float
    r_max    : float

    Returns
    -------
    post : (N_r,)     coarse posterior
    fine : (10000,)   fine r grid
    ip   : (10000,)   interpolated posterior
    """
    fine = np.linspace(r_min, r_max, 10000)
    post = posterior_from_chi2(chi2_arr)
    ip   = upscale_posterior(post, r_grid, fine)
    peak = fine[np.argmax(ip)]
    h68  = compute_hdi(ip, fine, 0.68)
    h95  = compute_hdi(ip, fine, 0.95)
    mean = compute_first_moment(ip, fine)
    sig  = compute_second_moment_sigma(ip, fine)
    print(f"\n{'='*52}  [{label}]")
    print(f"  Peak  : {peak:.6f}")
    print(f"  Mean  : {mean:.6f}")
    print(f"  Sigma : {sig:.6f}")
    print(f"  68% HDI: [{h68[0]:.6f}, {h68[1]:.6f}]")
    print(f"  95% HDI: [{h95[0]:.6f}, {h95[1]:.6f}]")
    print(f"{'='*52}")
    return post, fine, ip


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_OPTIONS = {
    "binning": {
        "nside": 64, "nlb": 10, "is_Dell": True,
        "ell_bins_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    "tensor_grid": {"r_min": 0.0, "r_max": 0.01, "n_r": 200},
    "cov_ordering": "ell_major",  # "ell_major" (default) or "pair_major" — see HOW_TO_USE.txt
    "n_data": 125, "fsky": 0.70,
    "pairs_mode": "all",          # "all" (253 pairs) or "cross" (231 cross-freq only; hard exclusion)
    "offset_type": "lollipop",    # "zero" | "lollipop" | "mc"
    "sim_batch_size": 20, "n_iter_lollipop": 30,
    "output_dir": "outputs",
}


def build_config(pkl_path):
    """
    Build the internal config dict from an input pickle path.

    Loads the pickle, merges its optional "options" dict over
    DEFAULT_OPTIONS, and stores the loaded pickle in config["inputs"]
    so it is not read twice from disk.
    """
    import pickle as _pkl
    import copy as _copy

    if not isinstance(pkl_path, (str, _os.PathLike)):
        raise TypeError(
            "run_hl_chi2 takes exactly one argument: the path to the input "
            "pickle file, e.g. run_hl_chi2('/path/to/hl_inputs.pkl').\n"
            "All inputs (arrays, channel_names, options) must be defined "
            "inside the pickle — see HOW_TO_USE.txt."
        )
    if not _os.path.isfile(pkl_path):
        raise FileNotFoundError(f"input pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        payload = _pkl.load(f)

    config = _copy.deepcopy(DEFAULT_OPTIONS)
    user_opts = payload.get("options", {})
    _known = set(DEFAULT_OPTIONS)
    _unknown = set(user_opts) - _known
    if _unknown:
        raise KeyError(
            f"unknown keys in pickle 'options': {sorted(_unknown)}.\n"
            f"  Allowed: {sorted(_known)}"
        )
    for k, v in user_opts.items():
        if isinstance(DEFAULT_OPTIONS[k], dict):
            config[k].update(v)
        else:
            config[k] = v

    config["inputs"] = {"path": str(pkl_path), "data": payload}
    return config


def run_hl_chi2(pkl_path):
    """
    Single callable for the HL HM1×HM1 chi2 analysis.

    The ONLY supported input is the path to a pickle file containing
    everything: data arrays, channel_names, and (optionally) an
    "options" dict overriding DEFAULT_OPTIONS.  See HOW_TO_USE.txt.

    Parameters
    ----------
    pkl_path : str — path to the input pickle

    Returns
    -------
    dict with keys:
        chi2        (N_r, N_data)
        r_grid      (N_r,)
        ell         (N_ell,)
        n_pairs_used  int
        pairs_mode  str
        offset_type str
        offset      (253, N_ell)  — cloff used
        data        dict          — full output of prepare_data()
        config      dict          — resolved options actually used
    """
    from hl_io import load_inputs, prepare_data

    config = build_config(pkl_path)

    raw  = load_inputs(config)
    data = prepare_data(raw, config)

    offset_type = config.get("offset_type", "zero")
    _fsky   = config.get("fsky", 0.70)
    _n_iter = config.get("n_iter_lollipop", 30)

    if offset_type == "zero":
        cloff = np.zeros((data["N_pairs_HM1"], data["N_ell"]))
    elif offset_type == "lollipop":
        cloff = compute_lollipop_offset(
            data["cldata"], data["fiducial"], data["ell"], _fsky, _n_iter
        )
    elif offset_type == "mc":
        cloff = compute_mc_offset(data["DLcross_all"])
    else:
        raise ValueError(f"offset_type must be 'zero', 'lollipop', or 'mc'; got '{offset_type}'")

    print(f"\n=== run_hl_chi2 ===")
    print(f"  pairs_mode  = {data['pairs_mode']}  ({data['N_pairs_used']} pairs)")
    print(f"  offset_type = {offset_type}")
    print(f"  N_data      = {data['N_data']}  N_r={data['N_r']}  N_ell={data['N_ell']}")

    # runtime check: HL requires cldata + cloff > 0 everywhere
    _shifted = data["cldata"] + cloff[np.newaxis, :, :]
    _n_neg   = int((_shifted <= 0).sum())
    if _n_neg > 0:
        _frac = _n_neg / _shifted.size * 100
        print(f"  [WARN] {_n_neg} entries ({_frac:.3f}%) of (cldata+cloff) are non-positive "
              f"— HL approximation may break.  Consider offset_type='mc'.")
    else:
        print(f"  [OK] all (cldata+cloff) entries are positive", flush=True)

    sim_batch_size = config.get("sim_batch_size", 20)
    chi2 = compute_hl_chi2_charmlike(
        clth=data["full_model"],
        cldata=data["cldata"],
        cloff=cloff,
        clfidu=data["fiducial"],
        icov=data["icov_pm"],
        N_fields=data["N_fields"],
        sim_batch_size=sim_batch_size,
        custom_pair_idxs=data["custom_pair_idxs"],
    )

    return {
        "chi2":         chi2,
        "r_grid":       data["r_grid"],
        "ell":          data["ell"],
        "n_pairs_used": data["N_pairs_used"],
        "pairs_mode":   data["pairs_mode"],
        "offset_type":  offset_type,
        "offset":       cloff,
        "data":         data,
        "config":       config,
    }

