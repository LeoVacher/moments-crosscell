import sys
sys.path.append('./lib/')
import os
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
fsky = '0.5_P_402_10deg_high' # Fraction of sky for the raw mask
complexity = 'high_complexity' # Sky complexity. Should be 'baseline', 'medium_complexity' or 'high_complexity'
load = False # Load previous sims 
path = '/pscratch/sd/s/svinzl/B_modes_project/' #path for saving downgraded maps and power spectra. Use './' for local and '/pscratch/sd/s/svinzl/B_modes_project/' for shared directory
load_maps = True # Load already downgraded maps stored in path
save_maps = True # Save downgraded maps in path
FM_only = True # If True, use full mission maps for auto-spectra. Otherwise use half missions
HM_only = False # If True, use only half mission maps (all combinations between HM1 and HM2)
kw = ''
kw_maps = ''

field = 'QU'

if FM_only:
    kw += '_FM'
    kw_maps = '_FM'
    hm1, hm2 = 0, 0
elif HM_only:
    load_maps = True
    kw += '_HM'
else:
    hm1, hm2 = 1, 2

if field != 'QU':
    kw += '_'+field

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

if FM_only:
    Pathload = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/coadd_sims/e2e_noise/%s' % (complexity)
else:
    Pathload = '/global/cfs/cdirs/litebird/simulations/maps/E_modes_postptep/2ndRelease/mock_splits_coadd_sims/e2e_noise/%s' % (complexity)

fg_type = complexity[0]

# Load instrument

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()

freq = instr['frequencies']
beam = instr['beams']
if HM_only:
    freq = np.tile(freq, 2)
    beam = np.tile(beam, 2)

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
if HM_only:
    bands = np.tile(bands, 2)

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
wsp0 = []
for i in range(Nfreqs):
    for j in range(i, Nfreqs):
        wsp.append(sim.get_wsp(null, null, null, null, mask, b, purify='BB', beam1=Bls[i], beam2=Bls[j]))

        if field == 'B':
            f1 = nmt.NmtField(mask, None, spin=0, beam=Bls[i])
            f2 = nmt.NmtField(mask, None, spin=0, beam=Bls[j])
            w0 = nmt.NmtWorkspace()
            w0.compute_coupling_matrix(f1, f2, b)
            wsp0.append(w0)

# Initialize simulations

if load:
    if masking_strat == '':
        if lbin is None:
            DLcross = np.load(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, fsky, scale, Nlbin, fg_type, fg_type, kw))
        else:
            DLcross = np.load(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, fsky, scale, Nlbin, lbin, fg_type, fg_type, kw))
    else:
        if lbin is None:
            DLcross = np.load(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, masking_strat, scale, Nlbin, fg_type, fg_type, kw))
        else:
            DLcross = np.load(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, masking_strat, scale, Nlbin, lbin, fg_type, fg_type, kw))
    
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

path_maps = path+'maps/maps_downgraded_nside%s_e2e_%s%s/' % (nside, complexity, kw_maps)

k_downgrade = 0
if load_maps:
    while k_downgrade < N and os.path.isfile(f'{path_maps}{k_downgrade}.npy'):
        k_downgrade += 1

elif not os.path.exists(path_maps):
    #maps = np.zeros((N, 3, Nfreqs, 2, Npix))
    os.makedirs(path_maps)

# Compute simulations

for k in trange(k_ini, N):
    if k >= k_downgrade:
        if FM_only:
            maps_k = np.zeros((Nfreqs, 2, Npix))

            for i in range(Nfreqs):
                FM_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d.fits' % (k, bands[i], complexity, k), field=None)

                maps_k[i] = sim.downgrade_map(FM_i, nside_in=512, nside_out=nside)[1:]
        
        else:
            maps_k = np.zeros((3, Nfreqs, 2, Npix))
            # Downgrade Q and U maps for each frequency
            for i in range(Nfreqs):
                FM_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_full.fits' % (k, bands[i], complexity, k), field=None)
                HM1_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitA.fits' % (k, bands[i], complexity, k), field=None)
                HM2_i = hp.read_map(Pathload+'/%04d/coadd_maps_LB_%s_cmb_e2e_sims_fg_%s_wn_1f_binned_030mHz_%04d_splitB.fits' % (k, bands[i], complexity, k), field=None)
            
                maps_k[0,i] = sim.downgrade_map(FM_i, nside_in=512, nside_out=nside)[1:]
                maps_k[1,i] = sim.downgrade_map(HM1_i, nside_in=512, nside_out=nside)[1:]
                maps_k[2,i] = sim.downgrade_map(HM2_i, nside_in=512, nside_out=nside)[1:]

        if save_maps:
            np.save(f'{path_maps}{k}.npy', maps_k)

    else:
        maps_k = np.load(f'{path_maps}{k}.npy')
    
    # Compute cross-spectra
    if FM_only:
        if field == 'QU':
            DLcross[k] = sim.computecross(maps_k, maps_k, maps_k, maps_k, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

        elif field == 'B':
            maps_B = np.zeros((Nfreqs, Npix))
            for i in range(Nfreqs):
                maps_B[i] = hp.alm2map(hp.map2alm([0*maps_k[i, 0], maps_k[i, 0], maps_k[i, 1]])[2], nside)
            z = 0
            for i in range(Nfreqs):
                for j in range(i, Nfreqs):
                    f1 = nmt.NmtField(mask, [maps_B[i]], beam=Bls[i])
                    f2 = nmt.NmtField(mask, [maps_B[j]], beam=Bls[j])
                    DLcross[k, z] = wsp0[z].decouple_cell(nmt.compute_coupled_cell(f1, f2))
                    z += 1

        elif field == 'QBUB':
            maps_QBUB = np.zeros((Nfreqs, 2, Npix))
            for i in range(Nfreqs):
                alm_B = hp.map2alm([0*maps_k[i, 0], maps_k[i, 0], maps_k[i, 1]])[2]
                maps_QBUB[i] = hp.alm2map([alm_B, 0*alm_B, alm_B], nside)[1:]
            z = 0
            for i in range(Nfreqs):
                for j in range(i, Nfreqs):
                    f1 = nmt.NmtField(mask, maps_QBUB[i], beam=Bls[i], purify_b=True)
                    f2 = nmt.NmtField(mask, maps_QBUB[j], beam=Bls[j], purify_b=True)
                    DLcross[k, z] = wsp[z].decouple_cell(nmt.compute_coupled_cell(f1, f2))[3]
                    z += 1

    elif HM_only:
        maps_k = np.concatenate(maps_k[1:])
        DLcross[k] = sim.computecross(maps_k, maps_k, maps_k, maps_k, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)
    
    else:
        DLcross[k] = sim.computecross(maps_k[0], maps_k[0], maps_k[hm1], maps_k[hm2], wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    # Save
    if masking_strat == '':
        if lbin is None:
            np.save(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, fsky, scale, Nlbin, fg_type, fg_type, kw), DLcross)
        else:
            np.save(path+'power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, fsky, scale, Nlbin, lbin, fg_type, fg_type, kw), DLcross)
    else:
        if lbin is None:
            np.save(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, masking_strat, scale, Nlbin, fg_type, fg_type, kw), DLcross)
        else:
            np.save(path+'power_spectra/DLcross_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc_gaussbeam_bandpass%s.npy' % (nside, masking_strat, scale, Nlbin, lbin, fg_type, fg_type, kw), DLcross)