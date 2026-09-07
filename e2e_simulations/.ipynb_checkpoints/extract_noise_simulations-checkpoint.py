import sys
sys.path.append('./lib/')
import os
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

nside = 64
N = 500
load = False
path = '/pscratch/sd/s/svinzl/B_modes_project/'
FM_only = True
kw = ''

if FM_only:
    Pathload = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/noise_simulations/'
    kw += '_FM'
else:
    Pathload = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_noise_simulations/'

Npixs = hp.nside2npix(nside)

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
Nfreqs = len(freq)

telescopes = ['LFT', 'MFT', 'HFT']
channels = {'LFT': ['L1-040', 'L2-050', 'L1-060', 'L3-068', 'L2-068', 'L4-078','L1-078', 'L3-089', 'L2-089', 'L4-100', 'L3-119', 'L4-140'],
            'MFT': ['M1-100', 'M2-119', 'M1-140', 'M2-166', 'M1-195'],
            'HFT': ['H1-195', 'H2-235', 'H1-280', 'H2-337', 'H3-402']
           }

# Initialize maps

path_maps = path+'maps/e2e_noise_nside%s%s/' % (nside, kw)

k_ini = 0
if load:
    while k_ini < 0 and os.path.isfile(f'{path_maps}{k_ini}.npy'):
        k_ini += 1

elif not os.path.exists(path_maps):
    os.makedirs(path_maps)

# Extract and downgrade noise simulations

for k in trange(k_ini, N):
    if FM_only:
        maps_k = np.zeros((Nfreqs, 3, Npixs))
    else:
        maps_k = np.zeros((3, Nfreqs, 3, Npixs))
    
    i = 0
    for t in telescopes:
        for c in channels[t]:
            if FM_only:
                maps_k[i] = hp.ud_grade(hp.read_map(Pathload+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d.fits' % (t, c, t, c, k), field=None), nside_out=nside)
                
            else:
                maps_k[0, i] = hp.ud_grade(hp.read_map(Pathload+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_full.fits' % (t, c, t, c, k), field=None), nside_out=nside)
                maps_k[1, i] = hp.ud_grade(hp.read_map(Pathload+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_splitA.fits' % (t, c, t, c, k), field=None), nside_out=nside)
                maps_k[2, i] = hp.ud_grade(hp.read_map(Pathload+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_splitB.fits' % (t, c, t, c, k), field=None), nside_out=nside)
            
            i += 1

    np.save(f'{path_maps}{k}.npy', maps_k)