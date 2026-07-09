#!/usr/bin/env python3
"""
make_input_pickle.py
====================
Assembles a pickle file for the HL likelihood from .npy input files.

QUICK START — the only section you need to edit is "CONFIGURATION" below.
Everything marked with  # ← CHANGE  must be adapted to your dataset.

Usage:
    python make_input_pickle.py

Then run the likelihood with:
    python run_hm1.py hl_inputs.pkl
"""

import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_stats import chs2idx, idx2chs


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ←  only edit this section
# ═══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# 1. Channel names
#    This is the FIXED standard LiteBIRD HM1 ordering — do NOT change
#    it unless your data pipeline uses a different ordering.
#    Position in this list = field index used to compute pair indices.
#
#    !! CRITICAL: the ordering here must exactly match the ordering
#    used by the pipeline that produced DLcross, DLnoise, foreground
#    and covariance.  The code cannot detect a mismatch automatically.
#    When you run this script it prints the full pair→(ch_i, ch_j) map
#    — cross-check it against your pipeline before using the pickle.
# ------------------------------------------------------------------
CHANNEL_NAMES = [
    "LFT_040", "LFT_050", "LFT_060",
    "LFT_068a", "LFT_068b", "LFT_078a",
    "LFT_078b", "LFT_089a", "LFT_089b",
    "LFT_100",  "LFT_119",  "LFT_140",
    "MFT_100",  "MFT_119",  "MFT_140",
    "MFT_166",  "MFT_195a", "MFT_195b",
    "HFT_235",  "HFT_280",  "HFT_337",  "HFT_402",
]

# ------------------------------------------------------------------
# 2. Input file paths
#    Point each key to the corresponding .npy file on your machine.
#    Expected shapes:
#      DLcross    : (N_sims, N_pairs, N_ell)
#      DLnoise    : (N_sims, N_pairs, N_ell)  or  (N_pairs, N_ell)
#      foreground : (N_pairs, N_ell)
#      covariance : (N_pairs*N_ell, N_pairs*N_ell)   [ell-major]
#      tensor_grid: (N_r, N_ell)                     [row 0 = r=0]
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_INP  = os.path.join(_HERE, "inputs_local")   # example data shipped with Likelihood_grid/

SRC = {
    "DLcross":     os.path.join(_INP, "DLcross_hm1.npy"),    # ← CHANGE to your path
    "DLnoise":     os.path.join(_INP, "DLnoise_hm1.npy"),    # ← CHANGE
    "foreground":  os.path.join(_INP, "fg_hm1.npy"),         # ← CHANGE
    "covariance":  os.path.join(_INP, "cov_hm1.npy"),        # ← CHANGE
    "tensor_grid": os.path.join(_INP, "tensor_grid.npy"),    # ← CHANGE
}

# ------------------------------------------------------------------
# 3. Output pickle path
# ------------------------------------------------------------------
OUTPUT = os.path.join(_HERE, "hl_inputs.pkl")      # ← CHANGE to your desired output path

# ------------------------------------------------------------------
# 4. Analysis options  (passed to the likelihood; change as needed)
# ------------------------------------------------------------------
OPTIONS = {
    "binning": {
        "nside":         64,                        # ← CHANGE if different resolution
        "nlb":           10,                        # ← CHANGE: ell bins per bandpower
        "is_Dell":       True,                      # True = D_ell, False = C_ell
        # List of bin indices (0-based) to keep.
        # Remove 0 to cut the lowest bin, remove 11 to cut the highest, etc.
        "ell_bins_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],   # ← CHANGE
    },
    "tensor_grid": {
        "r_min": 0.0,                               # must be 0 (matches tensor_grid row 0)
        "r_max": 0.01,                              # ← CHANGE: upper edge of r grid
        "n_r":   200,                               # ← CHANGE: number of r points
    },
    "n_data":         125,                          # ← CHANGE: number of data sims used
    "fsky":           0.70,                         # ← CHANGE: sky fraction
    # Covariance matrix ordering — MUST match how you produced covariance.npy:
    #   "ell_major"  : flat = ell_bin * N_pairs + pair_index  (NaMaster default)
    #   "pair_major" : flat = pair_index * N_ell  + ell_bin   (skip reorder)
    "cov_ordering":   "ell_major",                  # ← CHANGE if your cov is pair-major
    # Which pairs enter the likelihood:
    #   "all"   → all 253 pairs (auto + cross)  — recommended
    #   "cross" → 231 cross-frequency pairs only (i≠j), excludes auto-spectra
    #   NOTE: "cross" is a hard exclusion, NOT a marginalisation over auto pairs.
    "pairs_mode":     "all",                        # ← CHANGE
    "offset_type":    "lollipop",                   # "lollipop" or "none"
    "sim_batch_size": 20,                           # sims processed at once (memory vs speed)
    "n_iter_lollipop":30,                           # lollipop Newton iterations
    "output_dir":     "outputs",                    # ← CHANGE: where results are written
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    N_HM1   = len(CHANNEL_NAMES)
    N_PAIRS = N_HM1 * (N_HM1 + 1) // 2   # 253

    arrays = _load_arrays(SRC)
    N_sims, _, N_ell = arrays["DLcross"].shape

    _validate(arrays, CHANNEL_NAMES, N_HM1, N_PAIRS, N_ell)

    payload = {
        "channel_names": CHANNEL_NAMES,
        "DLcross":       arrays["DLcross"],
        "DLnoise":       arrays["DLnoise"],
        "foreground":    arrays["foreground"],
        "covariance":    arrays["covariance"],
        "tensor_grid":   arrays["tensor_grid"],
        "options":       OPTIONS,
        "meta": {
            "n_channels":    N_HM1,
            "n_pairs":       N_PAIRS,
            "n_ell":         N_ell,
            "n_sims":        N_sims,
            "pair_ordering": "upper-triangular, k = chs2idx(i,j,N) = i*(2N-i+1)//2 + (j-i)",
            "cov_ordering":  "ell-major: flat = ell_bin * N_pairs + pair_index",
            "dl_units":      "muK^2  (D_ell = ell*(ell+1)/(2pi)*C_ell)",
            "source_files":  {k: os.path.basename(v) for k, v in SRC.items()},
        },
    }

    print(f"\nSaving pickle to {OUTPUT}...")
    with open(OUTPUT, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"  Done. Size = {os.path.getsize(OUTPUT)/1e6:.1f} MB")
    print("\nTo run the likelihood:\n    python run_hm1.py hl_inputs.pkl\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_arrays(src):
    print("Loading source files...")
    arrays = {}
    for key, path in src.items():
        a = np.load(path)
        arrays[key] = a
        print(f"  {key:12s}: {a.shape}")
    return arrays


def _validate(arrays, channel_names, N_HM1, N_PAIRS, N_ell):
    """Shape and consistency checks. Raises AssertionError on failure."""
    print("\nRunning ordering and shape checks...")

    DLcross    = arrays["DLcross"]
    DLnoise    = arrays["DLnoise"]
    foreground = arrays["foreground"]
    covariance = arrays["covariance"]
    tensor     = arrays["tensor_grid"]

    _, N_pairs_dl, _ = DLcross.shape

    # 1. Pair count matches channel list
    assert N_pairs_dl == N_PAIRS, (
        f"[ERROR] DLcross has {N_pairs_dl} pairs but {N_HM1} channels → expected {N_PAIRS}.\n"
        f"  Check that CHANNEL_NAMES has exactly {N_HM1} entries."
    )

    # 2. All arrays share the same N_ell
    n_ell_noise = DLnoise.shape[-1]
    n_ell_fg    = foreground.shape[1]
    n_ell_cov   = covariance.shape[0] // N_PAIRS
    n_ell_grid  = tensor.shape[1]
    assert n_ell_noise == N_ell, f"[ERROR] DLnoise N_ell={n_ell_noise} ≠ DLcross N_ell={N_ell}"
    assert n_ell_fg    == N_ell, f"[ERROR] foreground N_ell={n_ell_fg} ≠ DLcross N_ell={N_ell}"
    assert n_ell_cov   == N_ell, (
        f"[ERROR] covariance dim {covariance.shape[0]} not divisible into "
        f"{N_PAIRS} pairs × {N_ell} ell bins (got {n_ell_cov})."
    )
    assert n_ell_grid  == N_ell, f"[ERROR] tensor_grid N_ell={n_ell_grid} ≠ DLcross N_ell={N_ell}"
    print(f"  [OK] all arrays have N_ell = {N_ell}")

    # 3. Covariance is square and right size
    expected_cov_dim = N_PAIRS * N_ell
    assert covariance.shape == (expected_cov_dim, expected_cov_dim), (
        f"[ERROR] covariance shape {covariance.shape} ≠ ({expected_cov_dim}, {expected_cov_dim})\n"
        f"  Expected (N_pairs*N_ell, N_pairs*N_ell) = ({N_PAIRS}*{N_ell}, {N_PAIRS}*{N_ell})."
    )
    print(f"  [OK] covariance shape {covariance.shape}")

    # 4. Tensor grid starts at zero (r=0 row)
    assert np.allclose(tensor[0], 0.0), (
        f"[ERROR] tensor_grid row 0 (r_min) should be all zeros.\n"
        f"  Got: {tensor[0]}.\n"
        f"  The grid must be built with r_grid[0] = 0."
    )
    print(f"  [OK] tensor_grid[0] = 0 (r=0)")

    # 5. Covariance is symmetric
    sym_err = np.abs(covariance - covariance.T).max()
    assert sym_err < 1e-10, f"[ERROR] covariance is not symmetric: max|C-C^T| = {sym_err:.2e}"
    print(f"  [OK] covariance symmetric (max|C-C^T| = {sym_err:.2e})")

    # 6. Pair ordering spot-check: verify first and last pair labels
    p0_i, p0_j = idx2chs(0, N_HM1)
    pm_i, pm_j = idx2chs(N_PAIRS - 1, N_HM1)
    assert (p0_i, p0_j) == (0, 0), f"pair 0 should be (0,0), got ({p0_i},{p0_j})"
    assert (pm_i, pm_j) == (N_HM1-1, N_HM1-1), \
        f"last pair should be ({N_HM1-1},{N_HM1-1}), got ({pm_i},{pm_j})"
    print(f"  [OK] pair 0 = {channel_names[p0_i]}×{channel_names[p0_j]}")
    print(f"  [OK] pair {N_PAIRS-1} = {channel_names[pm_i]}×{channel_names[pm_j]}")

    # 7. Print the FULL pair→(ch_i, ch_j) map for manual verification
    #    The code cannot detect a channel-ordering mismatch automatically.
    #    Check that pair 0 = your first auto, pair 1 = your first cross, etc.
    print(f"\n  !! VERIFY: pair ordering implied by CHANNEL_NAMES ({N_PAIRS} pairs)")
    print(f"  Cross-check this against the pipeline that produced your .npy files:")
    for k in range(N_PAIRS):
        ii, jj = idx2chs(k, N_HM1)
        print(f"    pair {k:3d} → {channel_names[ii]:12s} × {channel_names[jj]}")
    print(f"  If this does not match your pipeline ordering, update CHANNEL_NAMES.")

    # 8. Fiducial (fg + mean noise) consistency with mean of DLcross simulations
    mean_dl  = DLcross.mean(axis=0)
    noise_mn = DLnoise.mean(axis=0) if DLnoise.ndim == 3 else DLnoise
    fiducial = foreground + noise_mn
    rel_diff = np.abs(mean_dl - fiducial) / (np.abs(fiducial) + 1e-30)
    worst_pair = int(np.argmax(rel_diff.max(axis=1)))
    worst_rel  = float(rel_diff[worst_pair].max())
    wi, wj = idx2chs(worst_pair, N_HM1)
    if worst_rel > 0.05:
        print(f"\n  [WARN] fiducial vs mean(DLcross): max rel diff = {worst_rel*100:.2f}%"
              f" at pair {worst_pair} ({channel_names[wi]}×{channel_names[wj]})")
        print(f"         foreground/noise and DLcross may not share the same pipeline.")
    else:
        print(f"\n  [OK] fiducial (fg+noise) vs mean(DLcross): max rel diff = {worst_rel*100:.3f}%")


if __name__ == "__main__":
    main()
