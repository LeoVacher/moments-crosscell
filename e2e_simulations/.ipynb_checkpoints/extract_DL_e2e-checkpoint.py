import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim


# Inputs

nside = 64 #HEALPix nside
N = 250  #number of sims
lmax = nside*3-1 #maximum multipole
scale = 10 #apodization scale in degrees
Nlbin = 10 #binning scheme of the Cls
fsky = 0.7 #fraction of sky for the raw mask
complexity = 'baseline'
kws = '' #keyword for the simulation
masking_strat='' #keywords for choice of mask. If '', use Planck mask

if masking_strat=='GWD': #masking strategy From Gilles Weyman Depres (test)
    kws = kws + '_maskGWD'

Npix = hp.nside2npix(nside) #number of pixels
path = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_coadd_sims/e2e_noise/%s' % (complexity)

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
beam = instr['beams']

Nfreqs = len(freq)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

telescopes = ['LFT', 'MFT', 'HFT']
channels = {'LFT': ['L1-040', 'L2-050', 'L1-060', 'L3-068', 'L2-068', 'L4-078','L1-078', 'L3-089', 'L2-089', 'L4-100', 'L3-119', 'L4-140'],
            'MFT': ['M1-100', 'M2-119', 'M1-140', 'M2-166', 'M1-195'],
            'HFT': ['H1-195', 'H2-235', 'H1-280', 'H2-337', 'H3-402']
           }

# Downgrade simulations and comupute cross-spectra

