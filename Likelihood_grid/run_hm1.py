#!/usr/bin/env python3
"""
run_hm1.py — HL HM1×HM1 likelihood, pickle-based runner.

Usage::

    python run_hm1.py hl_inputs.pkl

EVERYTHING is defined inside the pickle: data arrays, channel_names and
(optionally) an "options" dict with the analysis settings, including
"output_dir" (default: ./outputs, created automatically) — see
HOW_TO_USE.txt.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hl_stats import run_hl_chi2, print_stats, per_sim_posteriors
from hl_plots import plot_likelihood_inputs, plot_posterior_single, plot_model_vs_data


def main():
    # ============================================================
    #  COMMAND LINE
    # ============================================================
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(sys.argv[0])} /path/to/hl_inputs.pkl")

    pkl_path = sys.argv[1]
    if not os.path.isfile(pkl_path):
        sys.exit(f"[ERROR] pickle not found: {pkl_path}")

    # ============================================================
    #  RUN — the pickle path is the ONLY input
    # ============================================================
    results = run_hl_chi2(pkl_path)

    chi2    = results["chi2"]
    r_grid  = results["r_grid"]
    ell     = results["ell"]
    ell_min = int(ell[0])
    data    = results["data"]
    config  = results["config"]        # resolved options actually used
    label   = f"{results['pairs_mode']}_{results['offset_type']}"

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}", flush=True)

    # ============================================================
    #  DIAGNOSTIC PLOTS — what enters the likelihood
    # ============================================================
    print("\n=== Diagnostic plots ===", flush=True)
    plot_likelihood_inputs(data, config, output_dir)
    plot_model_vs_data(
        data["cldata"], data["fiducial"], ell,
        list(config["hm1_channels"].keys()), data["N_fields"], output_dir,
    )

    # ============================================================
    #  POSTERIOR
    # ============================================================
    print("\n=== Posterior ===", flush=True)
    R_MIN, R_MAX = config["tensor_grid"]["r_min"], config["tensor_grid"]["r_max"]
    fine = np.linspace(R_MIN, R_MAX, 10000)

    _, _, ip = print_stats(f"{label} | ellmin={ell_min}", chi2, r_grid, R_MIN, R_MAX)
    pps = per_sim_posteriors(chi2)

    plot_posterior_single(
        ip_main       = ip,
        pps           = pps,
        chi2          = chi2,
        r_grid        = r_grid,
        fine          = fine,
        n_pairs       = results["n_pairs_used"],
        n_data        = data["N_data"],
        ell_bins_keep = config["binning"]["ell_bins_keep"],
        output_dir    = output_dir,
        ell_min       = ell_min,
        label         = label,
    )

    # ============================================================
    #  SAVE
    # ============================================================
    tag = f"{label}_ellmin{ell_min}"
    np.save(f"{output_dir}/chi2_{tag}.npy",      chi2)
    np.save(f"{output_dir}/posterior_{tag}.npy", ip)
    np.save(f"{output_dir}/r_grid.npy",          r_grid)
    print(f"\nSaved chi2_{tag}.npy, posterior_{tag}.npy, r_grid.npy to {output_dir}/")


if __name__ == "__main__":
    main()
