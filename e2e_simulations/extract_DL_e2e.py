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
masking_strat = 'intersection' # Masking strategy. Should be '', 'intersection' or 'union'
scale = 3 # Apodization scale in degrees
Nlbin = 10 # Binning scheme of the Cls
fsky = 0.7 # Fraction of sky for the raw mask
complexity = 'medium_complexity' # Sky complexity. Should be 'baseline', 'medium_complexity' or 'high_complexity'
load = False # Load previous sims 
path = '/pscratch/sd/s/svinzl/B_modes_project/' #path for saving downgraded maps and power spectra. Use './' for local and '/pscratch/sd/s/svinzl/B_modes_project/' for shared directory
load_maps = True # Load already downgraded maps stored in path
save_maps = False # Save downgraded maps in path

Npix = hp.nside2npix(nside) # Number of pixels
b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True) # Binning scheme for the cross-spectra
leff = b.get_effective_ells()
Nell = len(leff)
Pathload = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_coadd_sims/e2e_noise/%s' % (complexity) # Path to the simulations on the NERSC

fg_type = complexity[0]

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
beam = instr['beams']

Nfreqs = len(freq)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

Bls = np.zeros((Nfreqs, 3*nside))
for i in range(Nfreqs):
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
elif masking_strat == '':
    mask = hp.read_map(path+'masks/mask_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))
else:
    mask = hp.read_map(path+'masks/mask_%s_%s_nside%s_aposcale%s.npy' % (masking_strat, complexity, nside, scale))

# Initialize workspace

null = np.zeros((Nfreqs, 2, Npix))
wsp = []
for i in range(Nfreqs):
    for j in range(i, Nfreqs):
            wsp.append(sim.get_wsp(null, null, null, null, mask, b, purify='BB', beam1=Bls[i], beam2=Bls[j]))

# Initialize simulations

if load:
    if masking_strat == '':
        DLcross = np.load(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, Nlbin, fg_type, fg_type))
    else:
        DLcross = np.load(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, masking_strat, scale, Nlbin, fg_type, fg_type))
    
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

k_downgrade = 0
if load_maps:
    maps = np.load(path+'maps/maps_downgraded_nside%s_e2e_%s.npy' % (nside, complexity))
    while k_downgrade < N and np.any(maps[k_downgrade] != 0):
        k_downgrade += 1
else:
    maps = np.zeros((N, 3, Nfreqs, 2, Npix))

# Compute simulations

for k in trange(k_ini, N):
    if k >= k_downgrade:
        # Downgrade Q and U maps for each frequency
        for i in range(Nfreqs):
            FM_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_full.fits' % (k, bands[i], complexity, k), field=None)
            HM1_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitA.fits' % (k, bands[i], complexity, k), field=None)
            HM2_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitB.fits' % (k, bands[i], complexity, k), field=None)
        
            maps[k,0,i] = sim.downgrade_map(FM_i, nside_in=512, nside_out=nside)[1:]
            maps[k,1,i] = sim.downgrade_map(HM1_i, nside_in=512, nside_out=nside)[1:]
            maps[k,2,i] = sim.downgrade_map(HM2_i, nside_in=512, nside_out=nside)[1:]

        if save_maps:
            np.save(path+'maps/maps_downgraded_nside%s_e2e_%s.npy' % (nside, complexity), maps)

    # Compute cross-spectra
    DLcross[k] = sim.computecross(maps[k,0], maps[k,0], maps[k,1], maps[k,2], wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    # Save
    if masking_strat == '':
        np.save(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, fsky, scale, Nlbin, fg_type, fg_type), DLcross)
    else:
        np.save(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass.npy' % (nside, masking_strat, scale, Nlbin, fg_type, fg_type), DLcross)