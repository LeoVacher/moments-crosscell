import sys
sys.path.append('./lib')
import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import trange
import simu_lib as sim


# Inputs

nside = 64
N = 250
lmax = 3*nside-1
fsky = 0.7
scale = 10
Nlbin = 10
masking_strat = '' # Should be '' for Planck mask, 'GWD', 'intersection_<complexity>' or 'union_<complexity>'
load = False
path = '/pscratch/sd/s/svinzl/B_modes_project/'
gaussbeam = True
FM_only = False
HM_only = True
kw = ''

instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
beam = instr['beams']
if HM_only:
    beam = np.tile(beam, 2)

Nfreqs = len(beam)
Ncross = int(Nfreqs*(Nfreqs+1) / 2)

Bls = np.ones((Nfreqs, 3*nside))
if gaussbeam:
    kw += '_gaussbeam'
    for i in range(Nfreqs):
        Bls[i] = hp.gauss_beam(beam[i], lmax=3*nside-1, pol=True).T[2]

if FM_only:
    kw += '_FM'
    hm1, hm2 = 0, 0
elif HM_only:
    kw += '_HM'
else:
    hm1, hm2 = 1, 2

Npix = hp.nside2npix(nside)

# Load noise simulations

maps = np.load(path+'/maps/e2e_noise_nside%s.npy' % (nside))[:, :, :, 1:]

if masking_strat == '':
    mask = hp.read_map(path+'/masks/mask_fsky%s_nside%s_aposcale%s.npy'%(fsky, nside, scale))
else:
    mask = hp.read_map(path+'/masks/mask_%s_nside%s_aposcale%s.npy' % (masking_strat, nside, scale))

# Initialize workspace

b = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=Nlbin, is_Dell=True)
leff = b.get_effective_ells()
Nell = len(leff)

null = np.zeros((Nfreqs, 2, Npix))
if gaussbeam:
     wsp = []
     for i in range(Nfreqs):
         for j in range(i, Nfreqs):
            wsp.append(sim.get_wsp(null, null, null, null ,mask, b, purify='BB', beam1=Bls[i], beam2=Bls[j]))
else:
     wsp = sim.get_wsp(null, null, null, null, mask, b, purify='BB')

# Initialize simulations

if load:
    if masking_strat == '':
        DLnoise = np.load(path+'power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s%s.npy' % (nside, fsky, scale, Nlbin, kw))
    else:
        DLnoise = np.load(path+'power_spectra/DLnoise_nside%s_%s_scale%s_Nlbin%s%s.npy' % (nside, masking_strat, scale, Nlbin, kw))
    
    k_ini = np.argwhere(DLnoise == 0)[0,0]

    if k_ini == N:
        print('All sims already computed and saved')
        sys.exit()

    if k_ini > N:
        DLnoise_new = np.zeros((N, Ncross, Nell))
        DLnoise_new[:N,:,:] = DLgnilc[:N,:,:]
        DLnoise = DLgnilc_new

else:
    k_ini = 0
    DLnoise = np.zeros((N, Ncross, Nell))

# Compute simulations

for k in trange(k_ini, N):
    # Compute cross-spectra
    if not HM_only:
        DLnoise[k] = sim.computecross(maps[k,0], maps[k,0], maps[k,hm1], maps[k,hm2], wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    else:
        maps_k = np.concatenate(maps[k, 1:])
        DLnoise[k] = sim.computecross(maps_k, maps_k, maps_k, maps_k, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    # Save
    if masking_strat == '':
        np.save(path+'power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s%s_full.npy' % (nside, fsky, scale, Nlbin, kw), DLnoise)
    else:
        np.save(path+'power_spectra/DLnoise_nside%s_%s_scale%s_Nlbin%s%s.npy' % (nside, masking_strat, scale, Nlbin, kw), DLnoise)