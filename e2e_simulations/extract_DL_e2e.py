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
load = False # Load previous sims 

Npix = hp.nside2npix(nside) # Number of pixels
b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True) # Binning scheme for the cross-spectra
leff = b.get_effective_ells()
Nell = len(leff)
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

bands = []
for t in telescopes:
    for c in channels[t]:
        bands.append(t+'_'+c)

# Mask

if fsky == 1:
    mask = np.ones(Npix)
else:
    mask = hp.read_map('./masks/mask_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))

# Initialise workspace

null = np.zeros((3, Npix))
wsp = []
for i in range(Nfreqs):
    for j in range(i, Nfreqs):
            wsp.append(sim.get_wsp(null, null, null, null, mask, b, purify='BB', beam1=Bls[i], beam2=Bls[j]))

# Downgrade simulations and compute cross-spectra

if load == True:
    DLcross = np.load('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_e2e_%s.npy' % (nside, fsky, scale, Nlbin, complexity)) 
    k_ini = np.argwhere(DLcross == 0)[0,0]

    if k_ini == N:
        print('All sims already computed and saved')
        sys.exit()

    if k_ini > N:
        DLcross_new = np.zeros((N, Ncross, Nell))
        DLcross_new[:N,:,:] = DLcross[:N,:,:]
        DLcross = DLcross_new

else:
    k_ini = 0
    DLcross = np.zeros((N, Ncross, Nell))

for k in range(k_ini, N):
    maps_FM = np.zeros((Nfreqs, 3, Npix))
    maps_HM1 = np.zeros((Nfreqs, 3, Npix))
    maps_HM2 = np.zeros((Nfreqs, 3, Npix))

    for i in range(Nfreqs):
        FM_i = hp.read_map(path+'%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_full.fits' % (k, bands[i], complexity, k))
        HM1_i = hp.read_map(path+'%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitA.fits' % (k, bands[i], complexity, k))
        HM2_i = hp.read_map(path+'%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitB.fits' % (k, bands[i], complexity, k))

        maps_FM[i] = sim.downgrade_map(FM_i, nside_in=512, nside_out=nside)
        maps_HM1[i] = sim.downgrade_map(HM1_i, nside_in=512, nside_out=nside)
        maps_HM2[i] = sim.downgrade_map(HM2_i, nside_in=512, nside_out=nside)

    DLcross[k] = sim.computecross(maps_FM, maps_FM, maps_HM1, maps_HM2, wsp, mask, Nell, coupled=False, mode='BB', beams=Bls)

    np.save('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_e2e_%s.npy' % (nside, fsky, scale, Nlbin, complexity), DLcross)