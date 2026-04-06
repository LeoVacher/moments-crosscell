import sys
sys.path.append('./lib')
import numpy as np
import pymaster as nmt
from tqdm import trange
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import basicfunc as func
import analys_lib as an
import fitlib as ftl
import plotlib as plib
import covlib as cvl


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
HVTWD = False
gnilc = False
kwv = '_v1' # should be '' or '_v1'
betabar = 1.48
tempbar = 19.6
betasbar = -3.1
nu0d = 402
nu0s = 40
FM_only = False
HM_only = False

if gaussbeam:
    kws += '_gaussbeam'
if bandpass:
    kws += '_bandpass'
if FM_only:
    kws += '_FM'
if HM_only:
    kws += '_HM'

if adaptative:
    kw += '_adaptative'

# Instrument

instr_name = 'litebird_full'
instr = np.load('./lib/instr_dict/%s.npy'%instr_name, allow_pickle=True).item()
freq = instr['frequencies']
if HM_only:
    freq = np.tile(freq, 2)
N_freqs = len(freq)
Ncross = int(N_freqs * (N_freqs+1)/2)

if bandpass:
    bw = instr['bandwidths']
    if HM_only:
        bw = np.tile(bw, 2)
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

if FM_only:
    noise = np.load(Pathload+'/power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s_gaussbeam_FM' % (nside, fsky, scale, Nlbin)+'.npy')[:, :, :Nbins]
    data -= np.mean(noise, axis=0)
if HM_only:
    noise = np.load(Pathload+'/power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s_gaussbeam_HM' % (nside, fsky, scale, Nlbin)+'.npy')[:, :, :Nbins]
    data -= np.mean(noise, axis=0)

comp = [['A', 'As', 'Asd', 'Aw1b', 'Aw1t', 'w1bw1b', 'w1tw1t', 'w1bw1t', 'Asw1bs', 'w1bsw1bs', 'Asw1b', 'Asw1t', 'Aw1bs', 'w1bw1bs', 'w1tw1bs', 'cmb']
       for i in range(Nbins)]

gauss = an.gauss_like(freq, leff, covmat, comp, betabar, tempbar, betasbar, nu0d, nu0s)
results = gauss.run(data, n_iter=n_iter, adaptative=adaptative, pl_moms=pl_moms, HVTWD=HVTWD)

# Tensor-to-scalar ratio

Dl_lens, Dl_tens = ftl.getDL_cmb(nside=nside, Nlbin=Nlbin, cmb_e2e=cmb_e2e)
results['r'] = ((results['cmb'].T - Dl_lens) / Dl_tens).T

r, sigma_r = plib.getr_analytical(results)
print(f'\nGaussian approximation for r:\n'+
      f'bias:       {r}\n'+
      f'sigma:      {sigma_r}\n'+
      f'bias/sigma: {np.abs(r/sigma_r)}'
     )

# Noise residuals
'''
data_noise = np.load(Pathload+'/power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s_gaussbeam_HM_full.npy' % (nside, fsky, scale, Nlbin))[:, :, :Nbins]
results_noise = gauss.maximize(data_noise)
results_noise['r'] = (results_noise['cmb'].T / Dl_tens).T
'''
# Marginalization template

if gnilc:
    data_gnilc_fgres = np.load(Pathload+'/power_spectra/DLgnilc_fgres%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (kwv, nside, fsky, scale, Nlbin, dusttype, synctype)+'.npy')[:, :, :Nbins]
    results_gnilc_fgres = gauss.maximize(data_gnilc_fgres)
    results_gnilc_fgres['r'] = (results_gnilc_fgres['cmb'].T / Dl_tens).T

    data_gnilc_n = np.load(Pathload+'/power_spectra/DLgnilc_n%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc' % (kwv, nside, fsky, scale, Nlbin, dusttype, synctype)+'.npy')[:, :, :Nbins]
    results_gnilc_n = gauss.maximize(data_gnilc_n)
    results_gnilc_n['r'] = (results_gnilc_n['cmb'].T / Dl_tens).T

# Save results

np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw), results)
#np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_noise_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw), results_noise)

if gnilc:
    np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_gnilc_fgres%s_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw, kwv), results_gnilc_fgres)
    np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_gnilc_n%s_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw, kwv), results_gnilc_n)

"""
# Fits with CMB + noise

data_cmb = np.load(Pathload+'/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_c' % (nside, fsky, scale, Nlbin)+kws+'.npy')[:, :, :Nbins]

covmat_cmb = np.load(Pathload+'covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_c' % (cov_type, nside, fsky, scale, Nlbin)+kws+'.npy')[:Ncross*Nbins, :Ncross*Nbins]
gauss.N_inv = cvl.inverse_covmat(covmat_cmb, Ncross=Ncross, neglect_corbins=False)
gauss.A = gauss.compute_mixing_matrix()
gauss.W = gauss.compute_weight_matrix()

results_cmb = gauss.maximize(data_cmb)
results_cmb['r'] = ((results_cmb['cmb'].T - Dl_lens) / Dl_tens).T

np.save('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s%s_ds_o1bts%s_cmb_analytical' % (dusttype, synctype, fsky, scale, Nlbin, cov_type, kws, kw), results_cmb)
"""
# Global fit no marg

total_cmb = results['cmb'].T
cov_cmb = np.cov(results['cmb'])
N_inv_cmb = np.linalg.inv(cov_cmb)

A_cmb = np.zeros((Nbins, 1))
A_cmb[:, 0] = Dl_tens

W_cmb = np.linalg.inv(A_cmb.T @ N_inv_cmb @ A_cmb) @ A_cmb.T @ N_inv_cmb

samp_nomarg = np.zeros((N, 2))
chi2r_nomarg = np.zeros(N)

for i in range(N):
    d_cmb = total_cmb[i] - Dl_lens
    s_cmb = W_cmb @ d_cmb
    
    res = d_cmb - A_cmb @ s_cmb
    dof = len(d_cmb) - len(s_cmb)
    
    samp_nomarg[i, 0] = s_cmb[0]
    chi2r_nomarg[i] = res.T @ N_inv_cmb @ res / dof

samp_nomarg[:, 1] = 100

r_global = np.mean(samp_nomarg[:, 0])
sigma_r_global = np.std(samp_nomarg[:, 0])

print(f'\nGlobal fit:\n'+
      f'r          = {np.round(r_global, 5)} +/- {np.round(sigma_r_global, 5)}\n'+
      f'bias/sigma = {np.abs(r_global/sigma_r_global)}\n'
     )

"""
# Marginalization template

tempFGRs_gnilc = np.mean(results_gnilc['cmb'], axis=1)

tempFGRs_402 = np.mean(data[:, -1], axis=0)
tempFGRs_402 = tempFGRs_402 / np.mean(tempFGRs_402) * np.mean(results['cmb'].T - Dl_lens)

fg = np.load('./best_fits/results_d%ss%s_%s_scale%s_Nlbin%s_%s_gaussbeam_bandpass_ds_o%s%s_fg_analytical.npy'
               % (dusttype, synctype, fsky, scale, Nlbin, cov_type, '1bts', kw), allow_pickle=True).item()
r_fg = fg['r']
tempFGRs_fg = np.mean(r_fg.T * Dl_tens, axis=0)

tempFGRs_ideal = np.mean(results['cmb'], axis=1) - Dl_lens

# 1 step marginalization

A_cmb = np.zeros((Nbins, 2))
A_cmb[:, 0] = Dl_tens
A_cmb[:, 1] = tempFGRs_ideal

W_cmb = np.linalg.inv(A_cmb.T @ N_inv_cmb @ A_cmb) @ A_cmb.T @ N_inv_cmb

samp = np.zeros((N, 2))
chi2r = np.zeros(N)

for i in range(N):
    d_cmb = total_cmb[i] - Dl_lens
    s_cmb = W_cmb @ d_cmb

    res = d_cmb - A_cmb @ s_cmb
    dof = len(d_cmb) - len(s_cmb)

    samp[i] = s_cmb
    chi2r[i] = res.T @ N_inv_cmb @ res / dof

mu = np.mean(samp, axis=0)
sigma = np.cov(samp.T)

def posterior(x, mu, sigma):
    k = len(x)
    return (2*np.pi)**(-k/2) / np.sqrt(np.linalg.det(sigma)) * np.exp(-1/2 * (x-mu).T @ np.linalg.inv(sigma) @ (x-mu))

def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * (x-mu)**2 / sigma**2)
                                       
N_grid = 100

r_grid = np.linspace(-2e-2, 2e-2, N_grid)
alpha_grid = np.linspace(-5, 6, N_grid)

likelihood = np.zeros((N_grid, N_grid))
for i in range(N_grid):
    for j in range(N_grid):
        x = np.array([r_grid[i], alpha_grid[j]])
        likelihood[i, j] = posterior(x, mu, sigma)

marg_lkl = np.sum(likelihood, axis=-1)
marg_lkl /= np.max(marg_lkl)

test_lkl = gaussian(r_grid, mu[0], np.sqrt(sigma[0,0]))
test_lkl /= np.max(test_lkl)

plt.figure(figsize=(6.4,4.8))
plt.plot(r_grid, marg_lkl)
plt.plot(r_grid, test_lkl, '--')
plt.savefig('marg_lkl.pdf')
plt.close()
"""
"""
# 2 steps marginalization

A_alpha = np.zeros((Nbins, 1))
A_alpha[:, 0] = tempFGRs_gnilc
W_alpha = np.linalg.inv(A_alpha.T @ N_inv_cmb @ A_alpha) @ A_alpha.T @ N_inv_cmb

A_r = np.zeros((Nbins, 1))
A_r[:, 0] = Dl_tens
W_r = np.linalg.inv(A_r.T @ N_inv_cmb @ A_r) @ A_r.T @ N_inv_cmb

res = np.zeros((N, Nbins))

r = np.zeros(N)
alpha = np.zeros(N)

for i in range(N):
    d = total_cmb[i] - Dl_lens
    s_alpha = W_alpha @ d

    res[i] = d - A_alpha @ s_alpha
    s = W_r @ res[i]

    alpha[i] = s_alpha[0]
    r[i] = s[0]
"""
"""
# Marginalization with Gaussian likelihood

samples = []
pars = ['r', 'alpha']
labels = [r'r', r'\alpha']
legend = ['PySM', '402 GHz', 'GNILC']

for t, tempFGRs in enumerate([tempFGRs_fg, tempFGRs_402, tempFGRs_gnilc]):
    A_cmb = np.zeros((Nbins, 2))
    A_cmb[:, 0] = Dl_tens
    A_cmb[:, 1] = tempFGRs
    
    W_cmb = np.linalg.inv(A_cmb.T @ N_inv_cmb @ A_cmb) @ A_cmb.T @ N_inv_cmb
    
    samp = np.zeros((N, 2))
    chi2r = np.zeros(N)
    
    for i in range(N):
        d_cmb = total_cmb[i] - Dl_lens
        s_cmb = W_cmb @ d_cmb
    
        res = d_cmb - A_cmb @ s_cmb
        dof = len(d_cmb) - len(s_cmb)
    
        samp[i] = s_cmb
        chi2r[i] = res.T @ N_inv_cmb @ res / dof
    
    r_margin, alpha = np.mean(samp, axis=0)
    sigma_r_margin, sigma_alpha = np.std(samp, axis=0)

    samples.append(MCSamples(samples=samp, names=pars, labels=labels, label=legend[t]))

samples.append(MCSamples(samples=samp_nomarg, names=pars, labels=labels, label='no marg'))

# Corner plot

g = plots.get_subplot_plotter()
g.settings.title_limit = 1
g.settings.title_limit_fontsize = 12
g.settings.axes_fontsize = 12
g.settings.axes_labelsize = 16
g.settings.solid_colors = ['lightgrey', 'darkorange', 'dodgerblue', 'darkgreen']

g.triangle_plot(samples, filled=True, markers={'r': 0, 'alpha': 1}, marker_args={'lw': 0.7, 'ls': (0, (12, 8)), 'color': 'black'}, param_limits={'r': (-2e-2, 2e-2), 'alpha': (-3, 5)})

plt.savefig('posterior.pdf')
"""