#!/usr/bin/env python3
"""
hl_io.py
========
Data loading and preparation for the HL HM1×HM1 likelihood.

Reads the input pickle and turns it into the arrays the chi2 engine needs.
Called internally by run_hl_chi2 — you normally don't need to touch this.

Public API::

    raw  = load_inputs(config)        # load arrays from the pickle
    data = prepare_data(raw, config)  # slice ell bins, build model, invert cov
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import pickle as _pickle

import numpy as np
import pymaster as nmt
from hl_stats import chs2idx, idx2chs


# ---------------------------------------------------------------------------
# Pair-selection helpers
# ---------------------------------------------------------------------------

def _auto_pair_indices(n_hm1):
    """0-based indices of AUTO (i==j) pairs in upper-triangular ordering."""
    return np.array([chs2idx(i, i, n_hm1) for i in range(n_hm1)], dtype=int)


def _cross_pair_indices(n_hm1):
    """0-based indices of CROSS (i<j) pairs in upper-triangular ordering."""
    return np.array(
        [chs2idx(i, j, n_hm1) for i in range(n_hm1) for j in range(i + 1, n_hm1)],
        dtype=int,
    )


def _pairs_for_mode(pairs_mode, n_hm1):
    """
    Return (pair_indices, custom_pair_idxs) for the requested pairs_mode.

    pair_indices      — which of the 253 pairs to use in the covariance
    custom_pair_idxs  — passed to the HL engine to select from X; None = use all

    Supported modes:
      "all"   — all N*(N+1)/2 pairs (auto + cross)
      "cross" — cross-frequency pairs only (i<j); hard exclusion, NOT marginalisation
    """
    if pairs_mode == "all":
        return np.arange(n_hm1 * (n_hm1 + 1) // 2, dtype=int), None
    elif pairs_mode == "cross":
        idx = _cross_pair_indices(n_hm1)
        return idx, idx
    else:
        raise ValueError(f"pairs_mode must be 'all' or 'cross'; got '{pairs_mode}'")


# ---------------------------------------------------------------------------
# Covariance utilities
# ---------------------------------------------------------------------------

def reorder_cov_ell_to_pair_major(cov_em, n_pairs, n_ell):
    """
    Permute a covariance matrix from ell-major to pair-major storage order.

    Ell-major  : flat = e * n_pairs + p
    Pair-major : flat = p * n_ell   + e

    Parameters
    ----------
    cov_em  : (n_ell * n_pairs, n_ell * n_pairs) ndarray
    n_pairs : int
    n_ell   : int

    Returns
    -------
    cov_pm : (n_pairs * n_ell, n_pairs * n_ell) ndarray
    """
    perm = np.array(
        [e * n_pairs + p for p in range(n_pairs) for e in range(n_ell)],
        dtype=int,
    )
    return cov_em[np.ix_(perm, perm)]


def _extract_sub_cov(cov_em, pair_indices, n_pairs_full, n_ell):
    """Extract ell-major sub-covariance for the given pair indices."""
    flat_idx = np.array(
        [e * n_pairs_full + p for e in range(n_ell) for p in pair_indices],
        dtype=int,
    )
    return cov_em[np.ix_(flat_idx, flat_idx)]


# ---------------------------------------------------------------------------
# load_inputs: read .npy files from disk
# ---------------------------------------------------------------------------

def load_inputs(config):
    """
    Load all raw .npy files as specified in config.

    Parameters
    ----------
    config : dict  (same structure as the config built in run_hm1.py)

    Returns
    -------
    dict with keys:
        DLcross_raw  (N_sims, 253, N_ell_file)
        DLnoise_raw  (N_sims, 253, N_ell_file) or (253, N_ell_file)
        fg_raw       (253, N_ell_file)
        cov_raw      (N_ell_file*253, N_ell_file*253)  ell-major
        grid_raw     (N_r_file, N_ell_file)
        n_ell_file   int
        n_sims       int
    """
    print("\n=== load_inputs ===", flush=True)

    # --- pickle-only input mode ---
    if "inputs" not in config:
        raise KeyError(
            "[ERROR] config['inputs'] missing.  The only supported input mode "
            "is a pickle file: use run_hl_chi2('/path/to/hl_inputs.pkl')."
        )
    if "data" in config["inputs"]:
        _d = config["inputs"]["data"]          # already loaded by build_config
    else:
        with open(config["inputs"]["path"], "rb") as _f:
            _d = _pickle.load(_f)
    DLcross_raw = np.asarray(_d["DLcross"])
    DLnoise_raw = np.asarray(_d["DLnoise"])
    fg_raw      = np.asarray(_d["foreground"])
    cov_raw     = np.asarray(_d["covariance"])
    grid_raw    = np.asarray(_d["tensor_grid"])

    # channel ordering: read from pickle
    if "channel_names" not in _d:
        raise KeyError(
            "[ERROR] pickle is missing 'channel_names'.\n"
            "  Regenerate the pickle with make_input_pickle.py."
        )
    pkl_channels = list(_d["channel_names"])
    n_hm1 = len(pkl_channels)
    # populate the caller's config so prepare_data/plots see the channels
    config["hm1_channels"] = {name: i for i, name in enumerate(pkl_channels)}
    print(f"  channels ({n_hm1}): {pkl_channels[0]} … {pkl_channels[-1]}  [from pickle]")

    n_pairs_hm1  = n_hm1 * (n_hm1 + 1) // 2
    n_ell_file   = DLcross_raw.shape[2] if DLcross_raw.ndim == 3 else None

    print(f"  expected pairs  : {n_hm1}*({n_hm1}+1)/2 = {n_pairs_hm1}")
    print(f"  DLcross : {DLcross_raw.shape}  → (N_sims, N_pairs, N_ell)")
    print(f"  DLnoise : {DLnoise_raw.shape}")
    print(f"  fg      : {fg_raw.shape}        → (N_pairs, N_ell)")
    print(f"  cov     : {cov_raw.shape}  → (N_ell*N_pairs, N_ell*N_pairs)")
    print(f"  grid    : {grid_raw.shape}      → (N_r, N_ell)", flush=True)

    # ---- shape checks ----
    assert DLcross_raw.shape[1] == n_pairs_hm1, (
        f"[ERROR] DLcross has {DLcross_raw.shape[1]} pairs but config has "
        f"{n_hm1} channels → expected {n_pairs_hm1} pairs.\n"
        f"  Check: (1) hm1_channels has the right number of entries, "
        f"(2) DLcross was cut to HM1-only pairs."
    )
    _np_fg  = fg_raw.shape[0]
    _np_cov = cov_raw.shape[0]
    assert _np_fg == n_pairs_hm1, (
        f"[ERROR] fg has {_np_fg} rows but expected {n_pairs_hm1} pairs."
    )
    assert _np_cov % n_pairs_hm1 == 0, (
        f"[ERROR] cov first dim ({_np_cov}) is not divisible by N_pairs ({n_pairs_hm1}).\n"
        f"  Expected shape (N_ell*{n_pairs_hm1}, N_ell*{n_pairs_hm1})."
    )
    _n_ell_from_cov = _np_cov // n_pairs_hm1
    assert DLcross_raw.ndim == 3, (
        f"[ERROR] DLcross must be 3-D (N_sims, N_pairs, N_ell), got shape {DLcross_raw.shape}."
    )
    assert DLcross_raw.shape[2] == _n_ell_from_cov, (
        f"[ERROR] DLcross has {DLcross_raw.shape[2]} ell bins but cov implies "
        f"{_n_ell_from_cov}.  All files must share the same binning."
    )
    assert fg_raw.shape[1] == DLcross_raw.shape[2], (
        f"[ERROR] fg has {fg_raw.shape[1]} ell bins but DLcross has {DLcross_raw.shape[2]}."
    )
    print(f"  [OK] all shapes consistent — {_n_ell_from_cov} ell bins, {n_pairs_hm1} pairs", flush=True)

    raw = {"DLcross_raw": DLcross_raw, "DLnoise_raw": DLnoise_raw,
           "fg_raw": fg_raw, "cov_raw": cov_raw, "grid_raw": grid_raw,
           "n_ell_file": DLcross_raw.shape[2], "n_sims": DLcross_raw.shape[0],
           "n_hm1": n_hm1, "n_pairs_hm1": n_pairs_hm1}
    return raw


# ---------------------------------------------------------------------------
# prepare_data: ell selection, model, pair selection, covariance
# ---------------------------------------------------------------------------

def prepare_data(raw, config):
    """
    Apply ell selection, build model and fiducial, select pairs according to
    config["pairs_mode"], extract and invert the covariance.

    Parameters
    ----------
    raw    : output of load_inputs()
    config : CONFIG dict

    Returns
    -------
    dict — all arrays needed by the HL engine and the plots.

    Ordering contract
    -----------------
    The HL engine (compute_hl_chi2_charmlike) builds N_HM1×N_HM1 matrices
    from ALL 253 pairs, then selects elements via custom_pair_idxs.
    Therefore cldata, fiducial, full_model always carry all 253 pairs.
    The icov_pm is sized for the selected pairs only.
    """
    print("\n=== prepare_data ===", flush=True)

    n_hm1       = raw.get("n_hm1", len(config["hm1_channels"]))
    n_pairs_hm1 = raw.get("n_pairs_hm1", n_hm1 * (n_hm1 + 1) // 2)

    _bc = config["binning"]
    _lmax   = 2 * _bc["nside"] - 1
    _nmtbin = nmt.NmtBin.from_lmax_linear(lmax=_lmax, nlb=_bc["nlb"], is_Dell=_bc["is_Dell"])
    ell_all = _nmtbin.get_effective_ells()

    ell_bins_keep = _bc.get("ell_bins_keep", None)
    if ell_bins_keep is None:
        ell_bins_keep = list(range(len(ell_all)))
    ell_keep = np.asarray(ell_bins_keep, dtype=int)
    ell      = ell_all[ell_keep]
    n_ell    = len(ell_keep)

    _tg    = config["tensor_grid"]
    n_r    = _tg["n_r"]
    r_grid = np.linspace(_tg["r_min"], _tg["r_max"], n_r)
    n_data = config.get("n_data", raw["n_sims"] // 2)

    # ---- slice to ell_keep ----
    DLcross_all = raw["DLcross_raw"][:, :, ell_keep]        # (N_sims, 253, n_ell)

    noise_raw = raw["DLnoise_raw"]
    if noise_raw.ndim == 3:
        noise_mean = noise_raw[:, :, ell_keep].mean(axis=0) # (253, n_ell)
    else:
        noise_mean = noise_raw[:, ell_keep]                  # (253, n_ell)

    fg   = raw["fg_raw"][:, ell_keep]                       # (253, n_ell)
    grid = raw["grid_raw"][:n_r, ell_keep]                  # (N_r, n_ell)

    # ---- AUTO noise positivity check ----
    # For AUTO spectra, noise_mean must be >= 0 (physical sanity).
    # Negative noise_mean breaks the positive-definiteness of the fiducial matrix.
    _auto_local = np.array([chs2idx(i, i, n_hm1) for i in range(n_hm1)], dtype=int)
    _noise_auto = noise_mean[_auto_local]          # (n_hm1, n_ell)
    _n_neg_noise = int((_noise_auto < 0).sum())
    if _n_neg_noise > 0:
        _ch_names_l = list(config["hm1_channels"].keys())
        _bad = [(i, float(_noise_auto[i].min()))
                for i in range(n_hm1) if (_noise_auto[i] < 0).any()]
        print(f"  [WARN] noise_mean < 0 for {_n_neg_noise} AUTO (pair, ell) entries:")
        for _i, _minv in _bad[:5]:
            print(f"         {_ch_names_l[_i]}×{_ch_names_l[_i]}: min = {_minv:.3e}")
        if len(_bad) > 5:
            print(f"         ... and {len(_bad)-5} more channels")
        print(f"         AUTO noise must be non-negative.  Check noise simulation "
              f"consistency or use a larger noise simulation set.", flush=True)
    else:
        print(f"  [OK] noise_mean ≥ 0 for all {n_hm1} AUTO pairs", flush=True)

    # ---- covariance: slice to ell_keep if needed ----
    cov_raw   = raw["cov_raw"]
    n_ell_cov = cov_raw.shape[0] // n_pairs_hm1
    if n_ell_cov > n_ell:
        flat_keep = np.array(
            [e_bin * n_pairs_hm1 + p for e_bin in ell_keep for p in range(n_pairs_hm1)],
            dtype=int,
        )
        cov_hm1 = cov_raw[np.ix_(flat_keep, flat_keep)]
    else:
        cov_hm1 = cov_raw   # already at the right size

    # ---- fiducial and full model (all 253 pairs — needed for matrix ops) ----
    fiducial   = fg + noise_mean                             # (253, n_ell)
    full_model = (
        grid[:, np.newaxis, :]             # (N_r, 1,   n_ell)
        + fg[np.newaxis, :, :]             # (1,   253, n_ell)
        + noise_mean[np.newaxis, :, :]     # (1,   253, n_ell)
    )                                       # (N_r, 253, n_ell)

    cldata = DLcross_all[:n_data]           # (n_data, 253, n_ell)

    # ---- pair selection ----
    pairs_mode = config.get("pairs_mode", "all")
    pair_indices, custom_pair_idxs = _pairs_for_mode(pairs_mode, n_hm1)
    n_pairs_used = len(pair_indices)

    # ---- covariance for selected pairs → pair-major → invert ----
    if pairs_mode == "all":
        cov_used = cov_hm1
    else:
        cov_used = _extract_sub_cov(cov_hm1, pair_indices, n_pairs_hm1, n_ell)

    cov_ordering = config.get("cov_ordering", "ell_major")
    if cov_ordering == "pair_major":
        cov_pm = cov_used
    elif cov_ordering == "ell_major":
        cov_pm = reorder_cov_ell_to_pair_major(cov_used, n_pairs_used, n_ell)
    else:
        raise ValueError(f"cov_ordering must be 'ell_major' or 'pair_major'; got '{cov_ordering}'")
    icov_pm = np.linalg.inv(cov_pm)

    # ---- quick consistency check: mean(cldata) vs fiducial ----
    # Uses only the N_data test sims (same as the likelihood), not all N_sims.
    # Two metrics: pull = (mean-fidu)/(σ/√N_data) [statistical], rel = |mean-fidu|/|fidu| [physical].
    # Large pull + large rel → real model mismatch.
    # Large pull + small rel → inflated by tiny σ/√N (cross pair, near-zero noise scatter).
    _mean_dl  = cldata.mean(axis=0)
    _sig_dl   = cldata.std(axis=0) / np.sqrt(cldata.shape[0]) + 1e-30
    _pull     = (_mean_dl - fiducial) / _sig_dl
    _rel_diff = np.abs(_mean_dl - fiducial) / (np.abs(fiducial) + 1e-30)
    _worst_pair = int(np.argmax(np.abs(_pull).max(axis=1)))
    _worst_pull = float(np.abs(_pull).max())
    _worst_rel  = float(_rel_diff[_worst_pair].max())
    _wi, _wj  = idx2chs(_worst_pair, n_hm1)
    _ch_names = list(config["hm1_channels"].keys())
    if _worst_pull > 3.0:
        _flag = "[WARN]" if _worst_rel > 0.01 else "[INFO]"
        print(f"  {_flag} mean(DLcross) vs fiducial: max pull = {_worst_pull:.1f} σ/√N, "
              f"max rel = {_worst_rel*100:.2f}%")
        print(f"         at pair {_worst_pair} ({_ch_names[_wi]}×{_ch_names[_wj]})")
        if _worst_rel > 0.01:
            print(f"         → REAL mismatch (>{_worst_rel*100:.1f}%): fg/noise not from the "
                  f"same pipeline as DLcross — check consistency.")
        else:
            print(f"         → rel diff is small ({_worst_rel*100:.3f}%): pull inflated by "
                  f"tiny σ/√N (cross pair, near-zero noise scatter). Likely harmless.")
    else:
        print(f"  [OK] mean(DLcross) vs fiducial: max pull = {_worst_pull:.2f} σ/√N, "
              f"max rel = {_rel_diff.max()*100:.3f}%")

    print(f"  ell          : {ell}")
    print(f"  n_ell        = {n_ell}  n_data={n_data}  n_r={n_r}")
    print(f"  pairs_mode   = {pairs_mode}  →  {n_pairs_used} pairs used")
    print(f"  icov_pm      : {icov_pm.shape}", flush=True)

    return {
        # scalars
        "N_fields":          n_hm1,
        "N_pairs_HM1":       n_pairs_hm1,
        "N_pairs_used":      n_pairs_used,
        "N_ell":             n_ell,
        "N_r":               n_r,
        "N_data":            n_data,
        "N_sims":            raw["n_sims"],
        "pairs_mode":        pairs_mode,
        "pair_indices":      pair_indices,       # selected pair indices in 253-vector
        "custom_pair_idxs":  custom_pair_idxs,   # passed to HL engine (None if "all")
        # arrays (all 253 pairs — needed by matrix operations)
        "ell":               ell,
        "r_grid":            r_grid,
        "DLcross_all":       DLcross_all,         # (N_sims, 253, n_ell)
        "cldata":            cldata,               # (n_data, 253, n_ell)
        "noise_mean":        noise_mean,           # (253, n_ell)
        "fg":                fg,                   # (253, n_ell)
        "grid":              grid,                 # (N_r, n_ell)
        "fiducial":          fiducial,             # (253, n_ell)
        "full_model":        full_model,           # (N_r, 253, n_ell)
        # covariance for selected pairs, pair-major
        "icov_pm":           icov_pm,
        # pair info (useful for plots)
        "auto_pair_indices":  _auto_pair_indices(n_hm1),
        "cross_pair_indices": _cross_pair_indices(n_hm1),
    }


