import sys
sys.path.append('./lib')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

nside = 64
N = 500
fsky = '0.5_P_402_10deg_high'
scale = 10
masking_strat = '' # Should be '' for Planck mask, 'GWD', 'intersection_<complexity>' or 'union_<complexity>'
path = '/pscratch/sd/s/svinzl/B_modes_project/'
FM_only = True
kw = ''

if FM_only == True:
    kw += '_FM'

if masking_strat == '':
    kwsave = '_fsky%s_nside%s_aposcale%s%s' % (fsky, nside, scale, kw)
elif masking_strat == 'GWD':
    kwsave = '_GWD_fsky%s_nside%s_aposcale%s%s' % (fsky, nside, scale, kw)
else:
    kwsave = '_%s_nside%s_aposcale%s%s' % (masking_strat, nside, scale, kw)

# Load noise simulations

path_maps = path+'maps/e2e_noise_nside%s%s/' % (nside, kw)

if fsky == 1:
    mask = np.ones(hp.nside2npix(nside))
elif masking_strat == '':
    mask = hp.read_map(path+'masks/mask_fsky%s_nside%s_aposcale%s.npy'%(fsky, nside, scale))
else:
    mask = hp.read_map(path+'masks/mask_%s_nside%s_aposcale%s.npy' % (masking_strat, nside, scale))

if FM_only:
    Nfreqs = len(np.load(f'{path_maps}0.npy'))
else:
    Nfreqs = len(np.load(f'{path_maps}0.npy')[0])

# Compute CL_noise for each simulation

b = nmt.NmtBin.from_nside_linear(nside, nlb=1)

f = nmt.NmtField(mask, None, spin=0)
wT = nmt.NmtWorkspace()
wT.compute_coupling_matrix(f, f, b)

f = nmt.NmtField(mask, None, spin=2, purify_e=True)
wE = nmt.NmtWorkspace()
wE.compute_coupling_matrix(f, f, b)

f = nmt.NmtField(mask, None, spin=2, purify_b=True)
wB = nmt.NmtWorkspace()
wB.compute_coupling_matrix(f, f, b)

CL_noise = np.zeros((N, Nfreqs, 3, 3*nside))

for k in trange(N):
    if FM_only:
        maps_k = np.load(f'{path_maps}{k}.npy')
    else:
        maps_k = np.load(f'{path_maps}{k}.npy')[0]
    
    for i in range(Nfreqs):
        f = nmt.NmtField(mask, [maps_k[i,0]])
        CL_noise[k,i,0] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wT)[0]))

        f = nmt.NmtField(mask, maps_k[i,1:], purify_e=True)
        CL_noise[k,i,1] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wE)[0]))

        f = nmt.NmtField(mask, maps_k[i,1:], purify_b=True)
        CL_noise[k,i,2] = np.concatenate((np.zeros(2), sim.compute_master(f, f, wB)[3]))

    np.save('./e2e_simulations/CL_noise%s.npy' % (kwsave), CL_noise)