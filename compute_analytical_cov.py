import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import pymaster as nmt
import scipy.linalg as LA
import sys
sys.path.append("./lib")
import numpy as np
import pymaster as nmt 
import time
from mpfit import mpfit
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import simu_lib as sim
import fitlib as fit
import covlib as cvl

r=0.
nside = 64
lmax = nside*3-1
scale = 10
Nlbin = 10
lbin = 35
fsky = '0.5_P_402_10deg_high'
dusttype = 12
synctype = 7
kw=''
use_nmt=True
mode_cov='BB'
masking_strat = ''
gaussbeam = True
bandpass = True
Ngrid = 50
path = '/pscratch/sd/s/svinzl/B_modes_project/' # Path for saving covariance matrix
cl_noise = './e2e_simulations/' # Path to noise power spectra. Use 'white' for a Gaussian white noise model
cmb_e2e = True # If True, use CMB lensing power spectrum from litebird end-to-end simulations
FM_only = True # If True, compute the covariance matrix considering auto-spectra of full mission maps, otherwise use cross-spectra of half missions
HM_only = False # If True, use all combinations of half mission maps (auto- and cross-spectra)
kw_fg = ''
kw_noise = ''

field = 'QU'

if field == 'B':
    mode_cov = 'TT'

if masking_strat == 'GWD':
     kw += '_maskGWD'

if lbin is None:
    b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
else:
    ell_ini = np.concatenate((np.arange(2, lbin), np.arange(lbin, lmax+1, Nlbin)))
    ell_end = np.concatenate((np.arange(2, lbin)+1, np.arange(lbin, lmax+1, Nlbin)+Nlbin))
    ell_end = ell_end[ell_end <= lmax+1]
    ell_ini = ell_ini[:len(ell_end)]
    b = nmt.NmtBin.from_edges(ell_ini, ell_end, is_Dell=True)
    b.lmax = lmax

leff = b.get_effective_ells()
Nell = len(leff)
instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
sens_P = instr['sens_P']
freq = instr['frequencies']
beam = instr['beams']
if HM_only:
    freq = np.tile(freq, 2)
    sens_P = np.tile(freq, 2)
    beam = np.tile(beam, 2)
#sens_P = sens_P[np.argsort(freq)]
#beam = beam[np.argsort(freq)]
#freq = np.sort(freq)
N_freqs=len(freq)
Ncross= int(N_freqs*(N_freqs+1)/2)
Npix = hp.nside2npix(nside)
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))

if fsky == 1:
    mask = np.ones(Npix)
elif masking_strat == 'GWD':
     mask = hp.read_map(path+"masks/mask_GWD_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))
elif masking_strat == '':
    mask = hp.read_map(path+"masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))
else:
    if dusttype == 1 and synctype == 1:
        complexity = 'baseline'
    elif dusttype == 10 and synctype == 5:
        complexity = 'medium_complexity'
    elif dusttype == 12 and synctype == 7:
        complexity = 'high_complexity'
    mask = hp.read_map(path+'masks/mask_%s_%s_nside%s_aposcale%s.npy' % (masking_strat, complexity, nside, scale))
fsky_eff = np.mean(mask**2)

Bls_EE = np.ones((N_freqs, 3*nside))
Bls_BB = np.ones((N_freqs, 3*nside))
if gaussbeam:
	kw += '_gaussbeam'
	for i in range(N_freqs):
    		Bls_EE[i], Bls_BB[i] = hp.gauss_beam(beam[i], lmax=3*nside-1, pol=True).T[1:3]

if bandpass:
    kw += '_bandpass'
    bw = instr['bandwidths']
    if HM_only:
        bw = np.tile(bw, 2)
    freq_grids = np.zeros((N_freqs, Ngrid))
    for i in range(N_freqs):
        freq_grids[i] = np.geomspace(freq[i]-bw[i]/2, freq[i]+bw[i]/2, Ngrid)
    freq = freq_grids

if FM_only:
    auto = True
    kw_fg += '_FM'
    kw_noise += '_FM'
elif HM_only:
    auto = True
    kw_fg += '_HM'
else:
    auto = False

#signal

if use_nmt == False:
    DLdc = np.load(path+"power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy')

#foreground
    
if not HM_only:
    try:
        mapfg = np.load('./covariances/mapfg_d%ss%s%s.npy' % (dusttype, synctype, kw))
    except:
        mapfg = sim.get_fg_QU(freq, nside, dusttype=dusttype, synctype=synctype)
        np.save('./covariances/mapfg_d%ss%s%s.npy' % (dusttype, synctype, kw), mapfg)
        
else:
    try:
        mapfg = np.load('./covariances/mapfg_d%ss%s%s.npy' % (dusttype, synctype, kw))
    except:
        mapfg = sim.get_fg_QU(freq[:int(N_freqs/2)], nside, dusttype=dusttype, synctype=synctype)
        np.save('./covariances/mapfg_d%ss%s%s.npy' % (dusttype, synctype, kw), mapfg)
    mapfg = np.tile(mapfg.T, 2).T
        

#get fg spectra
b_unbined=  nmt.NmtBin.from_lmax_linear(lmax=nside*3-1,nlb=1)

if field == 'B':
    f = nmt.NmtField(mask, None, spin=0)
    wspB = nmt.NmtWorkspace()
    wspB.compute_coupling_matrix(f, f, b)
    wspB_unbined = nmt.NmtWorkspace()
    wspB_unbined.compute_coupling_matrix(f, f, b_unbined)
    wspT = wspB
    wspE = wspB
    
else:
    wspE_unbined = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b_unbined,purify='EE')
    wspE = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b,purify='EE')
    wspB_unbined = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b_unbined,purify='BB')
    wspB = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b,purify='BB')
    wspT = wspE # not needed in this case

ell_unbined= np.arange(3*nside)

if field == 'QU':
    CL_fg_EE = sim.computecross(mapfg,mapfg,mapfg,mapfg,wsp=wspE_unbined,Nell=len(ell_unbined),mask=mask,b=b_unbined,coupled=True,mode='EE')
    CL_fg_BB = sim.computecross(mapfg,mapfg,mapfg,mapfg,wsp=wspB_unbined,Nell=len(ell_unbined),mask=mask,b=b_unbined,coupled=True,mode='BB')

elif field == 'B':
    mapfg_B = np.zeros((N_freqs, Npix))
    for i in range(N_freqs):
        mapfg_B[i] = hp.alm2map(hp.map2alm([0*mapfg[i, 0], mapfg[i, 0], mapfg[i, 1]])[2], nside)
    CL_fg_BB = np.zeros((Ncross, len(ell_unbined)))
    z = 0
    for i in range(N_freqs):
        for j in range(i, N_freqs):
            f1 = nmt.NmtField(mask, [mapfg_B[i]])
            f2 = nmt.NmtField(mask, [mapfg_B[j]])
            CL_fg_BB[z] = nmt.compute_coupled_cell(f1, f2)
            z += 1

elif field == 'QBUB':
    mapfg_QEUE = np.zeros((N_freqs, 2, Npix))
    mapfg_QBUB = np.zeros((N_freqs, 2, Npix))
    for i in range(N_freqs):
        alm_E, alm_B = hp.map2alm([0*mapfg[i, 0], mapfg[i, 0], mapfg[i, 1]])[1:]
        mapfg_QEUE[i] = hp.alm2map([alm_E, alm_E, 0*alm_E], nside)[1:]
        mapfg_QBUB[i] = hp.alm2map([alm_B, 0*alm_B, alm_B], nside)[1:]
        CL_fg_EE = sim.computecross(mapfg_QEUE, mapfg_QEUE, mapfg_QEUE, mapfg_QEUE, wsp=wspE_unbined, Nell=len(ell_unbined), mask=mask, b=b_unbined, coupled=True, mode='EE')
        CL_fg_BB = sim.computecross(mapfg_QBUB, mapfg_QBUB, mapfg_QBUB, mapfg_QBUB, wsp=wspB_unbined, Nell=len(ell_unbined), mask=mask, b=b_unbined, coupled=True, mode='BB')

#get noise spectra

CL_cross_noise_EE = np.zeros((Ncross,3*nside))
CL_cross_noise_BB = np.zeros((Ncross,3*nside))

if cl_noise == 'white':
	for i in range(N_freqs):
		cross = cvl.cross_index(i, i, N_freqs)
		CL_cross_noise_EE[cross] = 4*np.pi/Npix * sigpix[i]**2
		CL_cross_noise_BB[cross] = CL_cross_noise_EE[cross]
        
else:
    if masking_strat == '':
        Cl_noise = np.load(cl_noise+'CL_noise_fsky%s_nside%s_aposcale%s%s.npy' % (fsky, nside, scale, kw_noise))[:, :, 1:, 2:3*nside]#2*nside]
    elif masking_strat == 'GWD':
        Cl_noise = np.load(cl_noise+'CL_noise_GWD_fsky%s_nside%s_aposcale%s%s.npy' % (fsky, nside, scale, kw_noise))[:, :, 1:, 2:3*nside]#2*nside]
    else:
        Cl_noise = np.load(cl_noise+'CL_noise_%s_%s_nside%s_aposcale%s%s.npy' % (masking_strat, complexity, nside, scale, kw_noise))[:, :, 1:, 2:2*nside]
        
    if HM_only:
        Cl_noise = np.array([np.tile(Cl_noise[k].T, 2).T for k in range(len(Cl_noise))])
        
    Cl_noise_mean = np.mean(Cl_noise, axis=(0,2))
    Cl_noise_std = np.std(Cl_noise, axis=(0,2))

    for i in range(N_freqs):
        cross = cvl.cross_index(i, i, N_freqs)
        p = [{'value': 4*np.pi/Npix * sigpix[i]**2, 'fixed': 0, 'limited': [1,1], 'limits': [0,np.inf]}]
        fa = {'Cl_noise': Cl_noise_mean[i], 'sigma_Cl': Cl_noise_std[i]}
        m = mpfit(fit.chi2_Nl, parinfo=p, functkw=fa, quiet=1)
        CL_cross_noise_EE[cross, 2:] = m.params[0] * np.ones(3*nside-2)
        CL_cross_noise_BB[cross] = CL_cross_noise_EE[cross]

Nls_EE = np.zeros((Ncross, 3*nside))
Nls_BB = np.zeros((Ncross, 3*nside))
for i in range(N_freqs): 
    cross = cvl.cross_index(i, i, N_freqs)
    CL_cross_noise_EE[cross] /= Bls_EE[i]**2
    CL_cross_noise_BB[cross] /= Bls_BB[i]**2

    if field == 'B':
        coupled_noise_BB = wspB_unbined.couple_cell([CL_cross_noise_BB[cross]])[0]
        Nls_BB[cross] = coupled_noise_BB

    else:
        coupled_noise_EE = wspE_unbined.couple_cell([CL_cross_noise_EE[cross], np.zeros_like(CL_cross_noise_EE[cross]), np.zeros_like(CL_cross_noise_EE[cross]), CL_cross_noise_EE[cross]])[0]
        coupled_noise_BB = wspB_unbined.couple_cell([CL_cross_noise_BB[cross], np.zeros_like(CL_cross_noise_BB[cross]), np.zeros_like(CL_cross_noise_BB[cross]), CL_cross_noise_BB[cross]])[3]
        Nls_EE[cross] = coupled_noise_EE
        Nls_BB[cross] = coupled_noise_BB
        
if HM_only:
    Nls_EE *= 2
    Nls_BB *= 2

#get cmb spectra
if cmb_e2e == False:
    CLcmb_or = hp.read_cl(path+'power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE
else:
    CLcmb_or = hp.read_cl(path+'power_spectra/Cls_LiteBIRD_e2e_r0.fits') #TT EE BB
CL_lens_EE = CLcmb_or[1,:nside*3]
CL_lens_BB = CLcmb_or[2,:nside*3]

if field == 'B':
    coupled_cmb_BB = wspB_unbined.couple_cell([CL_lens_BB])[0]
    CL_cmb_BB = np.array([coupled_cmb_BB for i in range(N_freqs) for j in range(i, N_freqs)])

else:
    coupled_cmb_EE = wspE_unbined.couple_cell([CL_lens_EE, np.zeros_like(CL_lens_EE), np.zeros_like(CL_lens_EE), CL_lens_BB])[0]
    coupled_cmb_BB = wspB_unbined.couple_cell([CL_lens_EE, np.zeros_like(CL_lens_EE), np.zeros_like(CL_lens_EE), CL_lens_BB])[3]
    CL_cmb_EE = np.array([ coupled_cmb_EE for i in range(N_freqs) for j in range(i, N_freqs)]) 
    CL_cmb_BB = np.array([ coupled_cmb_BB for i in range(N_freqs) for j in range(i, N_freqs)]) 

if use_nmt==False:
    cov_sg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_signal=[np.zeros_like(DLdc[0,:,:Nell]), np.zeros_like(DLdc[0,:,:Nell]), DLdc[0,:,:Nell]], type='Knox_signal', output=mode_cov, progress=True)
    cov_an = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Knox-fg', auto=auto, output=mode_cov, progress=True)
    cov_anfg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Knox+fg', auto=auto, output=mode_cov, progress=True)

if use_nmt==True:
    if field != 'B':
        cov_an = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Nmt-fg', auto=auto, output=mode_cov, progress=True)
    else:
        cov_an = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[CL_cmb_BB/fsky_eff, np.zeros_like(CL_cmb_BB), np.zeros_like(CL_cmb_BB)], Cls_fg=[CL_fg_BB/fsky_eff, np.zeros_like(CL_fg_BB), np.zeros_like(CL_fg_BB)], Nls=[Nls_BB/fsky_eff, np.zeros_like(Nls_BB), np.zeros_like(Nls_BB)], type='Nmt-fg', auto=auto, output=mode_cov, progress=True)

if cmb_e2e:
    if dusttype == 1 and synctype == 1:
        dusttype, synctype = 'b', 'b'
    if dusttype == 10 and synctype == 5:
        dusttype, synctype = 'm', 'm'
    if dusttype == 12 and synctype == 7:
        dusttype, synctype = 'h', 'h'

kw += kw_fg

if field != 'QU':
    kw += '_'+field

if use_nmt==False:
    if masking_strat not in ['intersection', 'union']:
        if lbin is None:
            np.save(path+'covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
            np.save(path+'covariances/cov_signal_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_sg)
            np.save(path+'covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
        else:
            np.save(path+'covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_an)
            np.save(path+'covariances/cov_signal_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_sg)
            np.save(path+'covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_anfg)
    else:
        if lbin is None:
            np.save(path+'covariances/cov_Knox-fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
            np.save(path+'covariances/cov_signal_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_sg)
            np.save(path+'covariances/cov_Knox+fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
        else:
            np.save(path+'covariances/cov_Knox-fg_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_an)
            np.save(path+'covariances/cov_signal_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_sg)
            np.save(path+'covariances/cov_Knox+fg_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_anfg)

if use_nmt==True:
    if masking_strat not in ['intersection', 'union']:
        if dusttype == None and synctype == None:
            if lbin is None:
                np.save(path+'covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_c'%(nside,fsky,scale,Nlbin)+kw+'.npy',cov_an)
            else:
                np.save(path+'covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_c'%(nside,fsky,scale,Nlbin,lbin)+kw+'.npy',cov_an)
        else:
            if lbin is None:
                np.save(path+'covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
                #np.save(path+'covariances/cov_Nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
            else:
                np.save(path+'covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_an)
                #np.save(path+'covariances/cov_Nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_anfg)
    else:
        if lbin is None:
            np.save(path+'covariances/cov_Nmt-fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
            #np.save(path+'covariances/cov_Nmt+fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
        else:
            np.save(path+'covariances/cov_Nmt-fg_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_an)
            #np.save(path+'covariances/cov_Nmt+fg_nside%s_%s_scale%s_Nlbin%s_lbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,lbin,dusttype,synctype)+kw+'.npy',cov_anfg)