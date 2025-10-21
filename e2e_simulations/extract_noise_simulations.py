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
Nsims = 250

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

# Extract and downgrade noise simulations

maps = np.zeros((Nsims, Nfreqs, 2, Npixs))

for i in trange(Nsims):
    j = 0
    for t in telescopes:
        for c in channels[t]:
            maps[i, j] = hp.ud_grade(hp.read_map(path+'%s/%s/LB_%s_%s_binned_wn_1f_030mHz_%04d_full.fits' % (t, c, t, c, i), field=(1,2)), nside_out=Nside)
            j += 1

    np.save('./e2e_simulations/maps_noise_nside%s_full.npy' % (Nside), maps)