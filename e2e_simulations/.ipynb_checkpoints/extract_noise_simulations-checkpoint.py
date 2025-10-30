import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

path = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_noise_simulations/'
Nside = 64
N = 250
load = False

Npixs = hp.nside2npix(Nside)

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

k_ini = 0
if load:
    maps = np.load('./e2e_simulations/e2e_noise_nside%s.npy' % (Nside)) 
    while np.any(maps[k_ini] != 0):
        k_ini += 1

else:
    maps = np.zeros((N, 3, Nfreqs, 3, Npixs))

# Extract and downgrade noise simulations

for k in trange(k_ini, N):
    i = 0
    for t in telescopes:
        for c in channels[t]:
            maps[k,0,i] = hp.ud_grade(hp.read_map(path+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_full.fits' % (t, c, t, c, k), field=None), nside_out=Nside)
            maps[k,1,i] = hp.ud_grade(hp.read_map(path+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_splitA.fits' % (t, c, t, c, k), field=None), nside_out=Nside)
            maps[k,2,i] = hp.ud_grade(hp.read_map(path+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_splitB.fits' % (t, c, t, c, k), field=None), nside_out=Nside)
            i += 1

    np.save('./e2e_simulations/e2e_noise_nside%s.npy' % (Nside), maps)