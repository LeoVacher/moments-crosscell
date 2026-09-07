import sys
sys.path.append('./lib/')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim

# Inputs

nside = 64
N = 500
lmax = 3*nside-1
fsky = '0.5_P_402_10deg_high'
scale = 10
Nlbin = 10
lbin = 35
masking_strat = '' # Should be '' for Planck mask, 'GWD', 'intersection_<complexity>' or 'union_<complexity>'
load = False
path = '/pscratch/sd/s/svinzl/B_modes_project/'
gaussbeam = True
kw = ''

field = 'QU'

if field != 'QU':
    kw += '_'+field

Pathload = '/global/cfs/cdirs/litebird/simulations/LB_e2e_simulations/e2e_ns512/2ndRelease/'
telescope = 'HFT'
channel = 'H3-402'

band = telescope+'_'+channel

Npix = hp.nside2npix(nside)

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

Bl = hp.gauss_beam(beam[bands == band][0], lmax=3*nside-1, pol=True).T[2]

# Mask

if fsky == 1:
    mask = np.ones(Npix)
elif masking_strat == '':
    mask = hp.read_map(path+'masks/mask_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))
else:
    mask = hp.read_map(path+'masks/mask_%s_%s_nside%s_aposcale%s.npy' % (masking_strat, complexity, nside, scale))

# Initialize workspace

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

null = np.zeros((Nfreqs, 2, Npix))
if gaussbeam:
    w = sim.get_wsp(null, null, null, null ,mask, b, purify='BB', beam1=Bl, beam2=Bl)

    if field == 'B':
        f = nmt.NmtField(mask, None, spin=0, beam=Bl)
        w0 = nmt.NmtWorkspace()
        w0.compute_coupling_matrix(f, f, b)
        
else:
    w = sim.get_wsp(null, null, null, null, mask, b, purify='BB')

# Initialize cross-spectra

k_ini = 0
if load:
    if masking_strat == '':
        DLcmb = np.load(path+'power_spectra/DLcmb_nside%s_fsky%s_scale%s_Nlbin%s.npy' % (nside, fsky, scale, Nlbin))
    else:
        DLcmb = np.load(path+'power_spectra/DLcmb_nside%s_%s_scale%s_Nlbin%s.npy' % (nside, masking_strat, scale, Nlbin))
    
    k_ini = np.argwhere(DLnoise == 0)[0,0]

    if k_ini == N:
        print('All sims already computed and saved')
        sys.exit()

    if k_ini > N:
        DLcmb_new = np.zeros((N, Nell))
        DLcmb_new[:N,:,:] = DLcmb[:N,:,:]
        DLcmb = DLcmb_new

else:
    k_ini = 0
    DLcmb = np.zeros((N, Nell))

# Downgrade simulations and compute CMB BB spectrum

for k in trange(k_ini, N):
    cmb = sim.downgrade_map(hp.read_map(Pathload+'%s/%s/input_cmb/LB_%s_cmb_%04d.fits' % (telescope, channel, band, k), field=None) * 1e6, nside_in=512, nside_out=nside)

    if field == 'QU':
        f = nmt.NmtField(mask, cmb[1:], purify_b=True)
        DLcmb[k] = w.decouple_cell(nmt.compute_coupled_cell(f, f))[3]

    elif field == 'B':
        cmb_B = hp.alm2map(hp.map2alm([0*cmb[0], cmb[1], cmb[2]])[2], nside)
        f = nmt.NmtField(mask, [cmb_B], beam=Bl)
        DLcmb[k] = w0.decouple_cell(nmt.compute_coupled_cell(f, f))

    elif field == 'QBUB':
        alm_B = hp.map2alm([0*cmb[0], cmb[1], cmb[2]])[2]
        cmb_QBUB = hp.alm2map([alm_B, 0*alm_B, alm_B], nside)[1:]
        f = nmt.NmtField(mask, cmb_QBUB, beam=Bl, purify_b=True)
        DLcmb[k] = w.decouple_cell(nmt.compute_coupled_cell(f, f))[3]
                                
    # Save
    if masking_strat == '':
        np.save(path+'power_spectra/DLcmb_nside%s_fsky%s_scale%s_Nlbin%s%s.npy' % (nside, fsky, scale, Nlbin, kw), DLcmb)
    else:
        np.save(path+'power_spectra/DLcmb_nside%s_%s_scale%s_Nlbin%s%s.npy' % (nside, masking_strat, scale, Nlbin, kw), DLcmb)
