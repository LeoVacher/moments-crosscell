'''
Create input files before running BmodeLike on the fitted simulations
'''

import sys
sys.path.append('./lib/')
import os
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
import fitlib as fit

# Inputs

nside = 64
lmax = 2*nside-1
lbin = 35
Nlbin = 10
fsky = '0.5_P_402_10deg_high'
scale = 10
complexity = 'high_complexity' # Should be 'baseline', 'medium_complexity' or 'high_complexity'
order = '1bts'
fix = 1
adaptative = True
cov_type = 'Nmt-fg'
unbin = False
FM_only = True
kw = ''
kws = ''
kwf = ''

field = 'QU'

if FM_only:
    kws += '_FM'

if complexity == 'baseline':
    dusttype, synctype = 'b', 'b'
elif complexity == 'medium_complexity':
    dusttype, synctype = 'm', 'm'
elif complexity == 'high_complexity':
    dusttype, synctype = 'h', 'h'

if adaptative:
    kw += '_adaptative'

if field != 'QU':
    kwf += '_'+field

name = 'moments_%s_%s_%s_o%s%s%s' % (complexity, fsky, cov_type, order, kw, kwf) # Name of the config
if unbin:
    name += '_unbinned'

path = '../BmodeLike/inputs/%s' % (name)

# Load results

results = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass%s_ds_o%s%s_analytical%s.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, order, kw, kwf), allow_pickle=True).item()
del results['beta_d'], results['T_d'], results['beta_s'], results['chi2r']

keys = list(results.keys())

Dl_cmb = results['cmb'].T
Nsims = Dl_cmb.shape[0]

# Binning scheme

mask = hp.read_map('/pscratch/sd/s/svinzl/B_modes_project/masks/mask_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))

if lbin is None:
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True) # Binning scheme for the cross-spectra
else:
    ell_ini = np.concatenate((np.arange(2, lbin), np.arange(lbin, 3*nside, Nlbin)))
    ell_end = np.concatenate((np.arange(2, lbin)+1, np.arange(lbin, 3*nside, Nlbin)+Nlbin))
    ell_end = ell_end[ell_end <= 3*nside]
    ell_ini = ell_ini[:len(ell_end)]
    b = nmt.NmtBin.from_edges(ell_ini, ell_end, is_Dell=True)
    b.lmax = 3*nside-1

f = nmt.NmtField(mask, None, spin=2, purify_b=True)
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f, f, b)

leff = b.get_effective_ells()
leff = leff[leff + (Nlbin+1)/2 <= lmax]

Nbins = len(leff)
lmins = [int(b.get_ell_min(i)) for i in range(Nbins)]
lmaxs = [int(b.get_ell_max(i)) for i in range(Nbins)]

bpw = w.get_bandpower_windows()[3, :Nbins, 3, :lmax+1]

# Recovered CMB, systematic and statistical residuals

Dl_cmb_mean = np.mean(Dl_cmb, axis=0)

Dl_lens = bpw @ hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :lmax+1]
Dl_tens = bpw @ hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :lmax+1]
Dl_input = np.load('/pscratch/sd/s/svinzl/B_modes_project/power_spectra/DLcmb_nside%s_fsky%s_scale%s_Nlbin%s%s.npy' % (nside, fsky, scale, Nlbin, kwf))[:Nsims, :Nbins]

#statFGRs = Dl_cmb - Dl_cmb_mean
#sysFGRs = Dl_cmb_mean - Dl_lens

FGRs = Dl_cmb - Dl_input
sysFGRs = np.mean(FGRs, axis=0)
statFGRs = FGRs - sysFGRs

lmax_QML = lbin-1
#QML_cmb = np.zeros((Nsims, lmax_QML-1))
#QML_cmb[:] = Dl_lens[:lmax_QML-1]
QML_cmb = Dl_input[:, :lmax_QML-1]

# Foreground residuals template
"""
gnilc = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass%s_ds_o%s%s_gnilc_fgres_v1_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, order, kw), allow_pickle=True).item()

noise = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass%s_ds_o%s%s_gnilc_n_v1_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, order, kw), allow_pickle=True).item()

Dl_gnilc_fgres = gnilc['cmb'].T
Dl_gnilc_n = noise['cmb'].T

tempFGRs = Dl_gnilc_fgres - np.mean(Dl_gnilc_n, axis=0)
"""
# Save inputs

if not os.path.isdir(path):
	os.makedirs(path)

np.save(path+'/namaster_statFGRs.npy', statFGRs)
np.save(path+'/namaster_sysFGRs.npy', sysFGRs)
#np.save(path+'/namaster_tempFGRs.npy', tempFGRs)
np.save(path+'/namaster_total.npy', Dl_cmb)
np.save(path+'/qml_cmb.npy', QML_cmb)
hp.write_cl(path+'/beam_0TP_pixwin16.fits', np.ones((3, lmax+1)), overwrite=True)
np.save(path+'/namaster_workspace.npy', bpw)

print('Binned inputs saved in subfolder %s! Bin edges to put in the config file:\n' % (name))
print('nmtbin_lmins =', lmins)
print('nmtbin_lmaxs =', lmaxs)
"""
# Plot spectra

Cl_sysFGRs = sysFGRs / leff / (leff+1) * 2*np.pi
Cl_tempFGRs = tempFGRs / leff / (leff+1) * 2*np.pi

Cl_lens = Dl_lens / leff / (leff+1) * 2*np.pi
Cl_tens = Dl_tens / leff / (leff+1) * 2*np.pi

plt.figure(figsize=(6.4, 4.8))
plt.fill_between(leff, 1e-3*Cl_tens, 1e-2*Cl_tens, color='lightgrey', label=r'$r \in [10^{-3}, 10^{-2}]$')
plt.plot(leff, Cl_lens, linestyle=(0, (10,6)), color='black', label='CMB lensing')
plt.plot(leff[Cl_sysFGRs>0], Cl_sysFGRs[Cl_sysFGRs>0], 'o', alpha=0.7, label=r'sysFGRs, $>0$')
plt.plot(leff[Cl_sysFGRs<0], -Cl_sysFGRs[Cl_sysFGRs<0], 'o', alpha=0.7, label=r'sysFGRs, $<0$')
plt.plot(leff, np.abs(np.mean(Cl_tempFGRs, axis=0)), label='abs(tempFGRs)')
plt.xlim(min(leff), max(leff))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\mathcal{C}_\ell^{BB}$')
plt.legend()
plt.savefig('./e2e_simulations/input_spectra.pdf')
"""