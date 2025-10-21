import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim


# Inputs

nside = 64 # HEALPix nside
N = 250  # Number of sims
lmax = nside*3-1 # Maximum multipole
scale = 10 # Apodization scale in degrees
Nlbin = 10 # Binning scheme of the Cls
fsky = 0.7 # Fraction of sky for the raw mask
complexity = 'baseline' # Sky complexity. Should be 'baseline', 'medium_complexity' or 'high_complexity'
kws = '' # Keyword for the simulation
masking_strat = '' # Keywords for choice of mask. If '', use Planck mask

if masking_strat=='GWD': # Masking strategy From Gilles Weyman Depres (test)
    kws = kws + '_maskGWD'

Npix = hp.nside2npix(nside) # Number of pixels
path = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_coadd_sims/e2e_noise/%s' % (complexity) # Path to the simulations on the NERSC

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
beam = instr['beams']

Nfreqs = len(freq)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

Bls = np.zeros((N_freqs, 3*nside))
for i in range(N_freqs):
    Bls[i] = hp.gauss_beam(beam[i], lmax=3*nside-1, pol=True).T[2]

telescopes = ['LFT', 'MFT', 'HFT']
channels = {'LFT': ['L1-040', 'L2-050', 'L1-060', 'L3-068', 'L2-068', 'L4-078','L1-078', 'L3-089', 'L2-089', 'L4-100', 'L3-119', 'L4-140'],
            'MFT': ['M1-100', 'M2-119', 'M1-140', 'M2-166', 'M1-195'],
            'HFT': ['H1-195', 'H2-235', 'H1-280', 'H2-337', 'H3-402']
           }

# Downgrade simulations and comupute cross-spectra


