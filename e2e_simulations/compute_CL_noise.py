import sys
sys.path.append('./lib')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

Nside = 64
Nsims = 250
fsky = 0.7
scale = 3
masking_strat = 'union_high_complexity' # Should be '' for Planck mask, 'GWD', 'intersection_<complexity>' or 'union_<complexity>'

if masking_strat == '':
    kwsave = '_fsky%s_nside%s_aposcale%s' % (fsky, Nside, scale)
elif masking_strat = 'GWD':
    kwsave = '_GWD_fsky%s_nside%s_aposcale%s' % (fsky, Nside, scale)
else:
    kwsave = '_%s_nside%s_aposcale%s' % (masking_strat, Nside, scale)

# Load noise simulations

maps = np.load('./e2e_simulations/e2e_noise_nside%s.npy' % (Nside))[:,0]

if masking_strat == '':
    mask = hp.read_map('./masks/mask_fsky%s_nside%s_aposcale%s.npy'%(fsky, Nside, scale))
else:
    mask = hp.read_map('./masks/mask_%s_nside%s_aposcale%s.npy' % (masking_strat, Nside, scale))

Nfreqs = maps.shape[1]

# Compute CL_noise for each simulation

b = nmt.NmtBin.from_nside_linear(Nside, nlb=1)

f = nmt.NmtField(mask, None, spin=0)
wT = nmt.NmtWorkspace()
wT.compute_coupling_matrix(f, f, b)

f = nmt.NmtField(mask, None, spin=2, purify_e=True)
wE = nmt.NmtWorkspace()
wE.compute_coupling_matrix(f, f, b)

f = nmt.NmtField(mask, None, spin=2, purify_b=True)
wB = nmt.NmtWorkspace()
wB.compute_coupling_matrix(f, f, b)

CL_noise = np.zeros((Nsims, Nfreqs, 3, 3*Nside))

for k in trange(Nsims):
    for i in range(Nfreqs):
        f = nmt.NmtField(mask, [maps[k,i,0]])
        CL_noise[k,i,0] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wT)[0]))

        f = nmt.NmtField(mask, maps[k,i,1:], purify_e=True)
        CL_noise[k,i,1] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wE)[0]))

        f = nmt.NmtField(mask, maps[k,i,1:], purify_b=True)
        CL_noise[k,i,2] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wB)[3]))

    np.save('./e2e_simulations/CL_noise%s.npy' % (kwsave), CL_noise)