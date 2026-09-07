import sys
sys.path.append('./lib')
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
FM_only = True
HM_only = False
kw = ''
kw_maps = ''

field = 'QU'

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
    kw_maps = '_FM'
elif HM_only:
    kw += '_HM'
else:
    hm1, hm2 = 1, 2

if field != 'QU':
    kw += '_'+field

Npix = hp.nside2npix(nside)

# Load noise simulations

path_maps = path+'/maps/e2e_noise_nside%s%s/' % (nside, kw_maps)

if masking_strat == '':
    mask = hp.read_map(path+'/masks/mask_fsky%s_nside%s_aposcale%s.npy'%(fsky, nside, scale))
else:
    mask = hp.read_map(path+'/masks/mask_%s_nside%s_aposcale%s.npy' % (masking_strat, nside, scale))

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
     wsp = []
     wsp0 = []
     for i in range(Nfreqs):
         for j in range(i, Nfreqs):
            wsp.append(sim.get_wsp(null, null, null, null ,mask, b, purify='BB', beam1=Bls[i], beam2=Bls[j]))

            if field == 'B':
                f1 = nmt.NmtField(mask, None, spin=0, beam=Bls[i])
                f2 = nmt.NmtField(mask, None, spin=0, beam=Bls[j])
                w0 = nmt.NmtWorkspace()
                w0.compute_coupling_matrix(f1, f2, b)
                wsp0.append(w0)
                
else:
     wsp = sim.get_wsp(null, null, null, null, mask, b, purify='BB')

# Initialize simulations

if load:
    if masking_strat == '':
        DLnoise = np.load(path+'power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s%s_full.npy' % (nside, fsky, scale, Nlbin, kw))
    else:
        DLnoise = np.load(path+'power_spectra/DLnoise_nside%s_%s_scale%s_Nlbin%s%s_full.npy' % (nside, masking_strat, scale, Nlbin, kw))
    
    k_ini = np.argwhere(DLnoise == 0)[0,0]

    if k_ini == N:
        print('All sims already computed and saved')
        sys.exit()

    if k_ini > N:
        DLnoise_new = np.zeros((N, Ncross, Nell))
        DLnoise_new[:N,:,:] = DLnoise[:N,:,:]
        DLnoise = DLnoise_new

else:
    k_ini = 0
    DLnoise = np.zeros((N, Ncross, Nell))

# Compute simulations

for k in trange(k_ini, N):
    maps_k = np.load(f'{path_maps}{k}.npy')[:, 1:]
    
    # Compute cross-spectra
    if FM_only:
        if field == 'QU':
            DLnoise[k] = sim.computecross(maps_k, maps_k, maps_k, maps_k, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

        elif field == 'B':
            maps_B = np.zeros((Nfreqs, Npix))
            for i in range(Nfreqs):
                maps_B[i] = hp.alm2map(hp.map2alm([0*maps_k[i, 0], maps_k[i, 0], maps_k[i, 1]])[2], nside)
            z = 0
            for i in range(Nfreqs):
                for j in range(i, Nfreqs):
                    f1 = nmt.NmtField(mask, [maps_B[i]], beam=Bls[i])
                    f2 = nmt.NmtField(mask, [maps_B[j]], beam=Bls[j])
                    DLnoise[k, z] = wsp0[z].decouple_cell(nmt.compute_coupled_cell(f1, f2))
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
                    DLnoise[k, z] = wsp[z].decouple_cell(nmt.compute_coupled_cell(f1, f2))[3]
                    z += 1

    elif HM_only:
        maps_k = np.concatenate(maps[k, 1:])
        DLnoise[k] = sim.computecross(maps_k, maps_k, maps_k, maps_k, wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    else:
        DLnoise[k] = sim.computecross(maps_k[0], maps_k[0], maps_k[hm1], maps_k[hm2], wsp, mask, Nell, b, coupled=False, mode='BB', beams=Bls)

    # Save
    if lbin is None:
        if masking_strat == '':
            np.save(path+'power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s%s_full.npy' % (nside, fsky, scale, Nlbin, kw), DLnoise)
        else:
            np.save(path+'power_spectra/DLnoise_nside%s_%s_scale%s_Nlbin%s%s_full.npy' % (nside, masking_strat, scale, Nlbin, kw), DLnoise)

    else:
        if masking_strat == '':
            np.save(path+'power_spectra/DLnoise_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s%s_full.npy' % (nside, fsky, scale, Nlbin, lbin, kw), DLnoise)
        else:
            np.save(path+'power_spectra/DLnoise_nside%s_%s_scale%s_Nlbin%s_lbin%s%s_full.npy' % (nside, masking_strat, scale, Nlbin, lbin, kw), DLnoise)