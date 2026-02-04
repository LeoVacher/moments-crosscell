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
lmax = 121
Nlbin = 10
fsky = 0.7
scale = 10
complexity = 'high_complexity' # Should be 'baseline', 'medium_complexity' or 'high_complexity'
order = '1bts'
fix = 1
adaptative = True
cov_type = 'Nmt-fg'
unbin = False
kw = ''

if complexity == 'baseline':
    dusttype, synctype = 'b', 'b'
elif complexity == 'medium_complexity':
    dusttype, synctype = 'm', 'm'
elif complexity == 'high_complexity':
    dusttype, synctype = 'h', 'h'

if adaptative:
    kw += '_adaptative'

name = 'moments_%s_%s_%s_o%s%s' % (complexity, fsky, cov_type, order, kw) # Name of the config
if unbin:
    name += '_unbinned'

path = '../BmodeLike/inputs/%s' % (name)

# Load results

results = np.load('./best_fits/results_d%ss%s_%s_scale%s_%s_gaussbeam_bandpass_ds_o%s_fix%s%s.npy'
               % (dusttype, synctype, fsky, scale, cov_type, order, fix, kw), allow_pickle=True).item()

keys = list(results.keys())
params = np.array([results[k] if k != 'T_d' else 1/results[k] for k in keys])

r = results['r'].T
Nsims = r.shape[0]

# Compute CMB BB spectrum for each simulation

b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=False)
leff = b.get_effective_ells()

if unbin:
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=1, is_Dell=False)
    ell = b.get_effective_ells()
    Nbins = len(ell)
    r = np.array([np.interp(ell, leff, r[i]) for i in range(Nsims)])
else:
    Nbins = len(leff)
    lmins = [int(b.get_ell_min(i)) for i in range(Nbins)]
    lmaxs = [int(b.get_ell_max(i)) for i in range(Nbins)]

Cl_lens = b.bin_cell(hp.read_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits')[2, :lmax+1])[:Nbins]
Cl_tens = b.bin_cell(hp.read_cl('./power_spectra/Cls_Planck2018_tensor_r1.fits')[2, :lmax+1])[:Nbins]

Cl_cmb = Cl_lens + r * Cl_tens

Cl_cmb_mean = np.mean(Cl_cmb, axis=0)

# Compute statistical foreground residuals

statFGRs = Cl_cmb - Cl_cmb_mean

# Compute systematic foreground residuals

sysFGRs = (Cl_cmb_mean - Cl_lens) * 0

# Compute foreground residuals template
'''
nu = np.array([353, 402])
dust = np.zeros((Nsims, Nbins))
tempFGRs = np.zeros((Nsims, Nbins))
for k in range(Nsims):
    for i in range(Nbins):
        if order == '0':
            dust[k, i] = fit.func_ds_o0(params[:, i, k], x1=nu, x2=nu, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens*0, DL_tens=Cl_tens*0)[1] / leff[i]/(leff[i]+1) * 2*np.pi
        elif order == '1bt':
            dust[k, i] = fit.func_ds_o1bt(params[:, i, k], x1=nu, x2=nu, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens*0, DL_tens=Cl_tens*0)[1] / leff[i]/(leff[i]+1) * 2*np.pi
        else:
            dust[k, i] = fit.func_ds_o1bts(params[:, i, k], x1=nu, x2=nu, nu0d=402, nu0s=40, ell=i, DL_lensbin=Cl_lens*0, DL_tens=Cl_tens*0)[1] / leff[i]/(leff[i]+1) * 2*np.pi
        
tempFGRs = dust / np.mean(dust) * np.mean(Cl_cmb - Cl_lens) #* np.mean((Cl_cmb_mean-Cl_lens) / dust)
'''

gnilc = np.load('./best_fits/results_d%ss%s_%s_scale%s_%s_gaussbeam_bandpass_ds_o%s_fix%s%s_gnilc.npy'
               % (dusttype, synctype, fsky, scale, cov_type, order, fix, kw), allow_pickle=True).item()
r_gnilc = gnilc['r'].T

if unbin:
    r_gnilc = np.array([np.interp(ell, leff, r_gnilc[i]) for i in range(50)])

tempFGRs = np.mean(r_gnilc * Cl_tens, axis=0)
#tempFGRs *= np.mean(Cl_cmb_mean - Cl_lens) / np.mean(tempFGRs)

#tempFGRs = Cl_cmb - Cl_lens

# Save inputs

if not os.path.isdir(path):
	os.makedirs(path)

np.save(path+'/namaster_statFGRs.npy', statFGRs)
np.save(path+'/namaster_sysFGRs.npy', sysFGRs)
np.save(path+'/namaster_tempFGRs.npy', tempFGRs)
np.save(path+'/namaster_total.npy', Cl_cmb)
np.save(path+'/namaster_total_0gauswin_spectra_CMB_QML.npy', Cl_cmb)
hp.write_cl(path+'/beam_0TP_pixwin16.fits', np.ones((3, Nbins)), overwrite=True)

if unbin:
    print('Unbinned inputs saved in subfolder %s!' % (name))
else:
    print('Binned inputs saved in subfolder %s! Bin edges to put in the config file:\n' % (name))
    print('nmtbin_lmins =', lmins)
    print('nmtbin_lmaxs =', lmaxs)
