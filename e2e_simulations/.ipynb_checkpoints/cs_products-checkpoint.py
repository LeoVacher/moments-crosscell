import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
from tqdm import trange
import fitlib as fit

# Inputs

nside = 64
lmax = 2*nside-1 #121
Nlbin = 10
fsky = 0.7
scale = 10
complexity = 'baseline' # Should be 'baseline', 'medium_complexity' or 'high_complexity'
order = '1bts'
fix = 1
adaptative = True
cov_type = 'Nmt-fg'
bp_grid = 50
kw = ''

if complexity == 'baseline':
    dusttype, synctype = 'b', 'b'
elif complexity == 'medium_complexity':
    dusttype, synctype = 'm', 'm'
elif complexity == 'high_complexity':
    dusttype, synctype = 'h', 'h'

if adaptative:
    kw += '_adaptative'

# Load instrument

instr_name = 'litebird_full' # Instrument name in ./lib/instr_dict/
instr = np.load("./lib/instr_dict/%s.npy" % instr_name, allow_pickle=True).item()
freq = instr['frequencies']
N_freqs = len(freq)
Ncross = int(N_freqs * (N_freqs+1)/2)

bw = instr['bandwidths']
freq_grids = np.zeros((N_freqs, bp_grid))
for i in range(N_freqs):
    freq_grids[i] = np.geomspace(freq[i]-bw[i]/2, freq[i]+bw[i]/2, bp_grid)
freq = freq_grids

nu1, nu2 = np.zeros((2, Ncross, bp_grid))
cross = 0
for i in range(N_freqs):
    for j in range(i, N_freqs):
        nu1[cross], nu2[cross] = freq[i], freq[j]
        cross += 1

# Load results

results = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
del results['beta_d'], results['T_d'], results['beta_s'], results['chi2r']

keys = list(results.keys())

# Foreground binned residuals

b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=False)
leff = b.get_effective_ells()
Nbins = len(leff)
lmins = np.array([int(b.get_ell_min(i)) for i in range(Nbins)])
lmaxs = np.array([int(b.get_ell_max(i)) for i in range(Nbins)])

r = results['r']
Nsims = r.shape[1]

Cl_lens = b.bin_cell(hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :lmax+1])
Cl_tens = b.bin_cell(hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :lmax+1])

fg_res = r.T * Cl_tens
np.save(f'./e2e_simulations/res_binned_{complexity}.npy', fg_res)
"""
# Noise residuals

results_noise = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_noise_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()

noise_res = results_noise['r'].T * Cl_tens
"""
"""
# Foreground unbinned residuals

b_unbinned = nmt.NmtBin.from_lmax_linear(lmax=lmaxs[-1], nlb=1, is_Dell=False)
ell = b_unbinned.get_effective_ells()
Nell = len(ell)

Cl_lens_unbinned = b_unbinned.bin_cell(hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :lmax+1])
Cl_tens_unbinned = b_unbinned.bin_cell(hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :lmax+1])

#bins = np.repeat(np.arange(Nbins), Nlbin) # Associate one bin to each multipole
#fg_res_unbinned = r[bins].T * Cl_tens_unbinned

r_interp = np.array([np.interp(ell, leff, r[:,i]) for i in range(Nsims)])
fg_res_unbinned = r_interp * Cl_tens_unbinned

np.save(f'./e2e_simulations/res_unbinned_{complexity}.npy', fg_res_unbinned)

# Total binned residuals

data = np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, Nlbin, dusttype, synctype))[:, :, :Nbins]

model = np.zeros_like(data)
if order == '0':
    model_func = fit.func_ds_o0
elif order == '1bt':
    model_func = fit.func_ds_o1bt
else:
    model_func = fit.func_ds_o1bts

for k in trange(Nsims, desc='Computing binned residuals'):
    for i in range(Nbins):
        model[k, :, i] = model_func(params[:, i, k], x1=nu1, x2=nu2, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens, DL_tens=Cl_tens)

tot_res = data - model

# Total unbinned residuals

data_unbinned = np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, 1, dusttype, synctype))[:, :, :Nell]

model_unbinned = np.zeros_like(data_unbinned)
for k in trange(Nsims, desc='Computing unbinned residuals'):
    for i in range(Nell):
        model_unbinned[k, :, i] = model_func(params[:, bins[i], k], x1=nu1, x2=nu2, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens_unbinned, DL_tens=Cl_tens_unbinned)

tot_res_unbinned = data_unbinned - model_unbinned
"""
"""
if order == '0':
    model_func = fit.func_ds_o0
elif order == '1bt':
    model_func = fit.func_ds_o1bt
else:
    model_func = fit.func_ds_o1bts

data = np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin1_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, dusttype, synctype))[:, :, :Nell]

params_interp = np.zeros((Npars, Nell, Nsims))
for k in range(Nsims):
    for i in range(Npars):
        params_interp[i, :, k] = np.interp(ell, leff, params[i, :, k])

b_DL = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=1, is_Dell=True)
DL_lens = b_DL.bin_cell(hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :lmax+1])
DL_tens = b_DL.bin_cell(hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :lmax+1])

fg = np.zeros_like(data)
for k in trange(Nsims, desc='Computing model'):
    for i in range(Nell):
        fg[k, :, i] = model_func(params_interp[:, i, k], x1=nu1, x2=nu2, nu0d=402, nu0s=40, ell=i, DL_lensbin=np.zeros(Nell), DL_tens=np.zeros(Nell))

fg_res_unbinned2 = (data - fg) / ell / (ell+1) * 2*np.pi - Cl_lens_unbinned

fg_402 = np.mean(fg[:, -1], axis=0) / ell / (ell+1) * 2*np.pi
tempFGRs = fg_402 / np.mean(fg_402) *  np.mean(fg_res_unbinned)

data_402 = np.mean(np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, 1, dusttype, synctype))[:, -1, :Nell], axis=0) / ell / (ell+1) * 2*np.pi
tempFGRs = data_402 / np.mean(data_402) *  np.mean(fg_res_unbinned)
"""


gnilc_fgres = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_gnilc_fgres_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
tempFGRs_gnilc = np.mean(gnilc_fgres['r'].T * Cl_tens, axis=0)

gnilc_n = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_gnilc_n_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
tempFGRs_noise = np.mean(gnilc_n['r'].T * Cl_tens, axis=0)


gnilc_fgres_v1 = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_gnilc_fgres_v1_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
tempFGRs_gnilc_v1 = np.mean(gnilc_fgres_v1['r'].T * Cl_tens, axis=0)

gnilc_n_v1 = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_gnilc_n_v1_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
tempFGRs_noise_v1 = np.mean(gnilc_n_v1['r'].T * Cl_tens, axis=0)

"""
r_gnilc_interp = np.array([np.interp(ell, leff, r_gnilc[:,i]) for i in range(50)])
tempFGRs_unbinned = np.mean(r_gnilc_interp * Cl_tens_unbinned, axis=0)
#tempFGRs_unbinned *= np.mean(fg_res_unbinned) / np.mean(tempFGRs_unbinned)
"""
"""
fg = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_fg_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
r_fg = fg['r']
tempFGRs_fg = np.mean(r_fg.T * Cl_tens, axis=0)
"""




data = np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (nside, fsky, scale, Nlbin, dusttype, synctype)+'_gaussbeam_bandpass'+'.npy')[:, :, :Nbins]
tempFGRs_402 = np.mean(data[:, -1], axis=0) / leff / (leff+1) * 2*np.pi
tempFGRs_402 = tempFGRs_402 / np.median(tempFGRs_402) * np.median(fg_res)

cmb = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_cmb_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, order, kw), allow_pickle=True).item()
r_cmb = cmb['r']
stat_cmb = r_cmb.T * Cl_tens

# Plot

plt.figure(figsize=(6.4, 4.8))
plt.plot(leff, 1e-2 * Cl_tens, linestyle='dotted', linewidth=1, color='black')
plt.plot(leff, 1e-3 * Cl_tens, linestyle='dotted', linewidth=1, color='black', label=r'$r \in \{10^{-2}, 10^{-3}\}$')
#plt.plot(leff, Cl_lens, linestyle=(0, (10,6)), linewidth=1, color='black', label='lensing')
plt.errorbar(leff, (np.mean(fg_res, axis=0)), yerr=np.std(fg_res, axis=0), fmt='o', label='fg residuals')
#plt.plot(ell, np.abs(np.mean(fg_res_unbinned, axis=0)), label='fg_res, unbinned')
#plt.plot(ell, np.abs(np.mean(fg_res_unbinned2[:, 0], axis=0)), label='|data-fg-lens|, 40x40')
#plt.plot(ell, np.abs(np.mean(fg_res_unbinned2[:, -1], axis=0)), label='|data-fg-lens|, 402x402')
#plt.plot(leff, np.abs(tempFGRs_gnilc - tempFGRs_noise), label='template v0')
#plt.plot(leff, np.abs(tempFGRs_gnilc_v1 - tempFGRs_noise_v1), label='template v1')
#plt.plot(leff, np.abs(tempFGRs_402), label='template from 402x402')
#plt.plot(leff, np.abs(np.mean(stat_cmb, axis=0)), '--', color='black', label='statistical variance')
#plt.plot(leff, np.abs(tempFGRs_fg), label='template from pysm')
#plt.plot(ell, np.abs(tempFGRs_unbinned), label='tempFGRs, unbinned')
#plt.plot(leff, np.abs(np.mean(tot_res[:, -1], axis=0)), label='total_res at 402 GHz, binned')
#plt.plot(ell, np.abs(np.mean(tot_res_unbinned[:, -1], axis=0)), label='total_res at 402 GHz, unbinned')
#plt.plot(leff, np.abs(np.mean(noise_res, axis=0)), '--', label='noise residuals')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\mathcal{C}_\ell^{\rm res}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./e2e_simulations/test.pdf')
plt.close()
"""
plt.figure(figsize=(6.4, 4.8))
plt.plot(leff, 1e-2 * Cl_tens, linestyle='dotted', linewidth=1, color='black')
plt.plot(leff, 1e-3 * Cl_tens, linestyle='dotted', linewidth=1, color='black', label=r'$r \in \{10^{-2}, 10^{-3}\}$')
plt.plot(leff, Cl_lens, linestyle=(0, (10,6)), linewidth=1, color='black', label='lensing')
plt.plot(leff, np.std(fg_res, axis=0), label='fg residuals')
plt.plot(leff, np.std(stat_cmb, axis=0), '--', color='black', label='statistical')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sigma(\mathcal{C}_\ell^{\rm res})$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./e2e_simulations/std.pdf')
plt.close()
"""