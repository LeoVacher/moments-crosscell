import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

Nside = 64
Nsims = 250

Npixs = hp.nside2npix(Nside)

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
Nfreqs = len(freq)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

# Load downgraded noise simulations

maps = np.load('./e2e_simulations/maps_noise_nside%s_full.npy' % (Nside))
mask = np.ones(Npixs)

# Compute DL_noise for each simulations

DL_noise = np.zeros((Nsims, Ncross, 2, 3*Nside-2))

for s in trange(Nsims):
    c = 0
    for i in range(Nfreqs):
        for j in range(i, Nfreqs):
            f1 = nmt.NmtField(mask, maps[s, i])
            f2 = nmt.NmtField(mask, maps[s, j])
            DL_noise[s, c] = nmt.compute_coupled_cell(f1, f2)[[0,3], 2:]
            
            c += 1

    np.save('./e2e_simulations/CL_noise_nside%s_full.npy' % (Nside), DL_noise)