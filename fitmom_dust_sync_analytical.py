import sys
sys.path.append('./lib')
import numpy as np
import pymaster as nmt
from tqdm import trange
import matplotlib.pyplot as plt
import basicfunc as func
import analys_lib as an
import fitlib as ftl
import plotlib as plib


# Inputs

nside = 64
lmax = 2*nside-1
fsky = 0.7
scale = 10
Nlbin = 10
dusttype = 'b'
synctype = 'b'
Pathload = '/pscratch/sd/s/svinzl/B_modes_project/'
N = 250
cov_type = 'Nmt-fg'
kw = ''
kws = ''
dusttype_cov = dusttype
synctype_cov = synctype
gaussbeam = True
bandpass = True
Ngrid = 50
cmb_e2e = True
n_iter = 3
adaptative = True
pl_moms = False
gnilc = True
betabar = 1.48
tempbar = 19.6
betasbar = -3.1
nu0d = 402
nu0s = 40
kw = ''

if gaussbeam:
    kws += '_gaussbeam'
if bandpass:
    kws += '_bandpass'

if adaptative:
    kw += '_adaptative'

# Instrument

instr_name = 'litebird_full'
instr = np.load('./lib/instr_dict/%s.npy'%instr_name, allow_pickle=True).item()
freq = instr['frequencies']
N_freqs = len(freq)
Ncross = int(N_freqs * (N_freqs+1)/2)

if bandpass:
    bw = instr['bandwidths']
    freq_grids = np.zeros((N_freqs, Ngrid))
    for i in range(N_freqs):
        freq_grids[i] = np.geomspace(freq[i]-bw[i]/2, freq[i]+bw[i]/2, Ngrid)
    freq = freq_grids

freq_pairs = np.array([(i, j) for i in range(N_freqs) for j in range(i, N_freqs)])
nu_i = freq[freq_pairs[:, 0]]
nu_j = freq[freq_pairs[:, 1]]

# Binning scheme

b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True)
leff = b.get_effective_ells()
Nbins = len(leff)

# Perform the fits

data = np.load(Pathload+'/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (nside, fsky, scale, Nlbin, dusttype, synctype)+kws+'.npy')[:, :, :Nbins]
covmat = np.load(Pathload+'covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (cov_type, nside, fsky, scale, Nlbin, dusttype_cov, synctype_cov)+kws+'.npy')[:Ncross*Nbins, :Ncross*Nbins]

comp = [['A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1bs', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs', 'cmb']
       for i in range(Nbins)]

gauss = an.gauss_like(freq, leff, covmat, comp, betabar, tempbar, betasbar, nu0d, nu0s)
results = gauss.run(data, n_iter=n_iter, adaptative=adaptative, pl_moms=pl_moms)

# Tensor-to-scalar ratio

Dl_lens, Dl_tens = ftl.getDL_cmb(nside=nside, Nlbin=Nlbin, cmb_e2e=cmb_e2e)
results['r'] = ((results['cmb'].T - Dl_lens) / Dl_tens).T

r, sigma_r = plib.getr_analytical(results)
print(f'\nGaussian approximation for r:\n'+
      f'bias:       {r}\n'+
      f'sigma:      {sigma_r}\n'+
      f'bias/sigma: {np.abs(r/sigma_r)}'
     )

if gnilc:
    data_gnilc = np.load(Pathload+'/power_spectra/DLgnilc_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (nside, fsky, scale, Nlbin, dusttype, synctype)+kws+'.npy')[:, :, :Nbins]
    results_gnilc = gauss.maximize(data_gnilc)
    results_gnilc['r'] = (results_gnilc['cmb'].T / Dl_tens).T

# Save results

np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw), results)

if gnilc:
    np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_gnilc_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw), results_gnilc)