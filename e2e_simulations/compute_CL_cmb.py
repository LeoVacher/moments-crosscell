import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

path = '/global/cfs/cdirs/litebird/simulations/LB_e2e_simulations/e2e_ns512/2ndRelease/'
Nside = 64
N = 500
load = False

# Instrument

telescopes = ['LFT', 'MFT', 'HFT']
channels = {'LFT': ['L1-040', 'L2-050', 'L1-060', 'L3-068', 'L2-068', 'L4-078','L1-078', 'L3-089', 'L2-089', 'L4-100', 'L3-119', 'L4-140'],
            'MFT': ['M1-100', 'M2-119', 'M1-140', 'M2-166', 'M1-195'],
            'HFT': ['H1-195', 'H2-235', 'H1-280', 'H2-337', 'H3-402']
           }

bands = []

for t in telescopes:
    for c in channels[t]:
        bands.append(t+'_'+c)

Nfreqs = len(bands)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

instr_name = 'litebird_full'
instr = np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
beam = instr['beams']

Bls = np.zeros((Nfreqs, 3*Nside))
for i in range(Nfreqs):
    Bls[i] = hp.gauss_beam(beam[i], lmax=3*Nside-1, pol=True).T[2]

# Initialize cross-spectra

k_ini = 0
if load:
    CL_cmb = np.load('./e2e_simulations/CL_cmb_nside%s.npy' % (Nside)) 
    while np.any(CL_cmb[k_ini] != 0):
        k_ini += 1

else:
    CL_cmb = np.zeros((N, Ncross, 3*Nside))

# Downgrade simulations and compute cross-spectra

Npixs = hp.nside2npix(Nside)
mask = np.ones(Npixs)

for k in trange(k_ini, N):
    cmb = np.zeros((Nfreqs, 2, Npixs))
    i = 0
    for t in telescopes:
        for c in channels[t]:
            cmb[i] = sim.downgrade_map(hp.read_map(path+'%s/%s/input_cmb/LB_%s_cmb_%04d.fits' % (t, c, bands[i], k), field=None) * 1e6, nside_in=512, nside_out=Nside)[1:]
            i += 1

    cross = 0
    for i in range(Nfreqs):
        for j in range(i, Nfreqs):
            f1 = nmt.NmtField(mask, cmb[i])
            f2 = nmt.NmtField(mask, cmb[j])
            CL_cmb[k, cross] = nmt.compute_coupled_cell(f1, f2)[3] / (Bls[i]*Bls[j])
            cross += 1
                                
    np.save('./e2e_simulations/CL_cmb_nside%s.npy' % (Nside), CL_cmb)

# Save mean lensing power spetrum

CL_lens = np.mean(CL_cmb, axis=(0,1))
hp.write_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.npy', CL_lens, overwrite=True)
