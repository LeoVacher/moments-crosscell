'''
Create input files before running BmodeLike on the fitted simulations
'''

import sys
sys.path.append('./lib/')
import os
import numpy as np
import healpy as hp
import pymaster as nmt
import fitlib as fit

# Inputs

nside = 64
lmax = 2*nside-1
Nlbin = 10
fsky = 0.7
complexity = 'baseline' # Should be 'baseline', 'medium_complexity' or 'high_complexity'
order = '1bts'
fix = 1
adaptative = True
cov_type = 'Nmt-fg'
kw = ''

if complexity == 'baseline':
    dusttype, synctype = 'b', 'b'
elif complexity == 'medium_complexity':
    dusttype, synctype = 'm', 'm'
elif complexity == 'high_complexity':
    dusttype, synctype == 'h', 'h'

if adaptative:
    kw += '_adaptative'

name = 'moments_%s_%s_%s_o%s%s' % (complexity, fsky, cov_type, order, kw) # Name of the config
path = '../BmodeLike/inputs/%s' % (name)

# Load results

results = np.load('./best_fits/results_d%ss%s_%s_%s_gaussbeam_bandpass_ds_o%s_fix%s%s.npy'
               % (dusttype, synctype, fsky, cov_type, order, fix, kw), allow_pickle=True).item()

keys = list(results.keys())

params = np.array([results[k] for k in keys])

# Compute CMB BB spectrum for each simulation

b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=False)
leff = b.get_effective_ells()
Nbins = len(leff)
lmins = [int(b.get_ell_min(i)) for i in range(Nbins)]
lmaxs = [int(b.get_ell_max(i)) for i in range(Nbins)]

r = results['r']
Nsims = r.shape[1]

Cl_lens = b.bin_cell(hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :2*nside])[:Nbins]
Cl_tens = b.bin_cell(hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :2*nside])[:Nbins]

Cl_cmb = np.zeros((Nsims, Nbins))
for k in range(Nsims):
    Cl_cmb[k] = Cl_lens + r[:, k] * Cl_tens

Cl_cmb_mean = np.mean(Cl_cmb, axis=0)

# Compute statistical foreground residuals

statFGRs = np.zeros((Nsims, Nbins))
for k in range(Nsims):
    statFGRs[k] = Cl_cmb[k] - Cl_cmb_mean

# Compute systematic foreground residuals

sysFGRs = Cl_cmb_mean - Cl_lens

# Compute foreground template

nu = np.array([402])

tempFGRs = np.zeros((Nsims, Nbins))
for k in range(Nsims):
    for i in range(Nbins):
        tempFGRs[k, i] = fit.func_ds_o1bts(params[:, i, k], x1=nu, x2=nu, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens*0, DL_tens=Cl_tens*0) / leff[i]/(leff[i]+1) * 2*np.pi

# Save inputs

if not os.path.isdir(path):
	os.makedirs(path)

np.save(path+'/namaster_statFGRs.npy', statFGRs)
np.save(path+'/namaster_sysFGRs.npy', sysFGRs)
np.save(path+'/namaster_tempFGRs.npy', tempFGRs)
np.save(path+'/namaster_total.npy', Cl_cmb)

print('Inputs saved in subfolder %s! Bin edges to put in the config file:\n' % (name))
print('nmtbin_lmins =', lmins)
print('nmtbin_lmaxs =', lmaxs)
