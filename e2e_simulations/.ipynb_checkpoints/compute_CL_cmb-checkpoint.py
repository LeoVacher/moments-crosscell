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
telescope = 'HFT'
channel = 'H3-402'

band = telescope+'_'+channel

# Instrument

telescopes = ['LFT', 'MFT', 'HFT']
channels = {'LFT': ['L1-040', 'L2-050', 'L1-060', 'L3-068', 'L2-068', 'L4-078','L1-078', 'L3-089', 'L2-089', 'L4-100', 'L3-119', 'L4-140'],
            'MFT': ['M1-100', 'M2-119', 'M1-140', 'M2-166', 'M1-195'],
            'HFT': ['H1-195', 'H2-235', 'H1-280', 'H2-337', 'H3-402']
           }

bands = np.array([])
for t in telescopes:
    for c in channels[t]:
        bands = np.append(bands, t+'_'+c)

Nfreqs = len(bands)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

instr_name = 'litebird_full'
instr = np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
beam = instr['beams']

Bls = hp.gauss_beam(beam[bands == band][0], lmax=3*Nside-1, pol=True).T

# Initialize cross-spectra

k_ini = 0
if load:
    CL_cmb = np.load('./e2e_simulations/CL_cmb_nside%s.npy' % (Nside)) 
    while np.any(CL_cmb[k_ini] != 0):
        k_ini += 1

else:
    CL_cmb = np.zeros((N, 3, 3*Nside))

# Downgrade simulations and compute cross-spectra

Npixs = hp.nside2npix(Nside)
mask = np.ones(Npixs)

for k in trange(k_ini, N):
    cmb = sim.downgrade_map(hp.read_map(path+'%s/%s/input_cmb/LB_%s_cmb_%04d.fits' % (telescope, channel, band, k), field=None) * 1e6, nside_in=512, nside_out=Nside)
            
    f = nmt.NmtField(mask, [cmb[0]])
    CL_cmb[k, 0] = nmt.compute_coupled_cell(f, f)[0] / Bls[0]**2

    f = nmt.NmtField(mask, cmb[1:])
    CL_cmb[k, 1:] = nmt.compute_coupled_cell(f, f)[[0,3]] / Bls[[1,2]]**2
                                
    np.save('./e2e_simulations/CL_cmb_nside%s.npy' % (Nside), CL_cmb)

# Save mean lensing power spetrum

CL_cmb_mean = np.zeros((3, 3*Nside))
for i in range(3):
    CL_cmb_mean[i] = np.mean(CL_cmb[:, i], axis=0)
    
hp.write_cl('./power_spectra/Cls_LiteBIRD_e2e_r0.fits', CL_cmb_mean, overwrite=True)
