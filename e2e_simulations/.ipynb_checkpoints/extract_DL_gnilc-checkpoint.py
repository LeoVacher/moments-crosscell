import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim


# Inputs

nside = 64 # HEALPix nside
N = 500  # Number of sims
lmax = nside*3-1 # Maximum multipole
masking_strat = '' # Masking strategy. Should be '', 'intersection' or 'union'
scale = 10 # Apodization scale in degrees
Nlbin = 10 # Binning scheme of the Cls
lbin = 35 # If not None, bin only multipoles > lbin
fsky = '0.5_402_high' # Fraction of sky for the raw mask
complexity = 'high' # Sky complexity. Should be 'baseline', 'medium' or 'high'
load = False # Load previous sims 
path = '/pscratch/sd/s/svinzl/B_modes_project/' # Path for saving power spectra. Use './' for local and '/pscratch/sd/s/svinzl/B_modes_project/' for shared directory
output = 'noise_residuals' # Compute the power spectra of total output maps or of noise residuals. Should be 'output_total' or 'noise_residuals'
v1 = True # Whether to use gnilc_maps_v1 or gnilc_maps.
FM_only = True
kw = ''

if FM_only:
    kw += '_FM'

if output == 'output_total':
    prefix = 'DLgnilc_fgres'
elif output == 'noise_residuals':
    prefix = 'DLgnilc_n'

if v1:
    kwv = '_v1'
else:
    N = min(N, 10)
    kwv = ''

Npix = hp.nside2npix(nside) # Number of pixels

if lbin is None:
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True) # Binning scheme for the cross-spectra
else:
    ell_ini = np.concatenate((np.arange(2, lbin), np.arange(lbin, lmax+1, Nlbin)))
    ell_end = np.concatenate((np.arange(2, lbin)+1, np.arange(lbin, lmax+1, Nlbin)+Nlbin))
    ell_end = ell_end[ell_end <= lmax+1]
    ell_ini = ell_ini[:len(ell_end)]
    b = nmt.NmtBin.from_edges(ell_ini, ell_end, is_Dell=True)
    b.lmax = lmax
    
leff = b.get_effective_ells()
Nell = len(leff)

Pathload = '/global/cfs/cdirs/litebird/results/b_modes_postptep/gnilc_maps%s/%s/%s' % (kwv, complexity, output) # Path to the GNILC maps on the NERSC

fg_type = complexity[0]

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']

Nfreqs = len(freq)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

beam = 70.5 / 60 * np.pi/180
Bls = hp.gauss_beam(beam, lmax=3*nside-1, pol=True).T[2] * hp.pixwin(nside, lmax=3*nside-1, pol=True)[1] * np.ones((Nfreqs, 3*nside))

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
        if lbin is None:
            DLgnilc = np.load(path+'power_spectra/%s%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, fsky, scale, Nlbin, fg_type, fg_type, kw))
        else:
            DLgnilc = np.load(path+'power_spectra/%s%s_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, fsky, scale, Nlbin, lbin, fg_type, fg_type, kw))
    else:
        if lbin is None:
            DLgnilc = np.load(path+'power_spectra/%s%s_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, masking_strat, scale, Nlbin, fg_type, fg_type, kw))
        else:
            DLgnilc = np.load(path+'power_spectra/%s%s_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, masking_strat, scale, Nlbin, lbin, fg_type, fg_type, kw))
    
    k_ini = np.argwhere(DLgnilc == 0)[0,0]

    if k_ini == N:
        print('All sims already computed and saved')
        sys.exit()

    if k_ini > N:
        DLgnilc_new = np.zeros((N, Ncross, Nell))
        DLgnilc_new[:N,:,:] = DLgnilc[:N,:,:]
        DLgnilc = DLgnilc_new

else:
    k_ini = 0
    DLgnilc = np.zeros((N, Ncross, Nell))

# Compute simulations

for k in trange(k_ini, N):
    if FM_only:
        # Extract Q and U maps for each frequency
        maps = np.zeros((Nfreqs, 2, Npix))
        for i in range(Nfreqs):
            maps[i] = hp.read_map(Pathload+'/%05d/QU_%s_LB_%s_70.5acm_ns64_lmax191_%05d.fits' % (k, output, bands[i], k), field=None)

        # Compute cross-spectra
        DLgnilc[k] = sim.computecross(maps, maps, maps, maps, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)
    
    else:
        # Extract Q and U maps for each frequency
        maps = np.zeros((3, Nfreqs, 2, Npix))
        for i in range(Nfreqs):
            maps[0, i] = hp.read_map(Pathload+'_splitAB/%05d/QU_%s_splitAB_LB_%s_70.5acm_ns64_lmax191_%05d.fits' % (k, output, bands[i], k), field=None)
            maps[1, i] = hp.read_map(Pathload+'_splitA/%05d/QU_%s_splitA_LB_%s_70.5acm_ns64_lmax191_%05d.fits' % (k, output, bands[i], k), field=None)
            maps[2, i] = hp.read_map(Pathload+'_splitB/%05d/QU_%s_splitB_LB_%s_70.5acm_ns64_lmax191_%05d.fits' % (k, output, bands[i], k), field=None)
    
        # Compute cross-spectra
        DLgnilc[k] = sim.computecross(maps[0], maps[0], maps[1], maps[2], wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    # Save
    if masking_strat == '':
        if lbin is None:
            np.save(path+'power_spectra/%s%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, fsky, scale, Nlbin, fg_type, fg_type, kw), DLgnilc)
        else:
            np.save(path+'power_spectra/%s%s_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, fsky, scale, Nlbin, lbin, fg_type, fg_type, kw), DLgnilc)
    else:
        if lbin is None:
            np.save(path+'power_spectra/%s%s_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, masking_strat, scale, Nlbin, fg_type, fg_type, kw), DLgnilc)
        else:
            np.save(path+'power_spectra/%s%s_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (prefix, kwv, nside, masking_strat, scale, Nlbin, lbin, fg_type, fg_type, kw), DLgnilc)