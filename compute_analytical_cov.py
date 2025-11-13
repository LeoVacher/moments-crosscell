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
scale = 3
Nlbin = 10
fsky = 0.7
dusttype = 1
synctype = 1
kw=''
use_nmt=True
mode_cov='BB'
masking_strat = 'union'
gaussbeam = True
bandpass = True
Ngrid = 100
path = '/pscratch/sd/s/svinzl/B_modes_project/' # Path for saving covariance matrix
cl_noise = './e2e_simulations/' # Path to noise power spectra. Use 'white' for a Gaussian white noise model
cmb_e2e = True # If True, use CMB lensing power spectrum from litebird end-to-end simulations

if masking_strat == 'GWD':
     kw += '_maskGWD'

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
leff = b.get_effective_ells()
Nell = len(leff)
instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
sens_P = instr['sens_P']
freq = instr['frequencies']
beam = instr['beams']
#sens_P = sens_P[np.argsort(freq)]
#beam = beam[np.argsort(freq)]
#freq = np.sort(freq)
N_freqs=len(freq)
Ncross= int(N_freqs*(N_freqs+1)/2)
Npix = hp.nside2npix(nside)
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))

if masking_strat == 'GWD':
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
    freq_grids = np.zeros((N_freqs, Ngrid))
    for i in range(N_freqs):
        freq_grids[i] = np.geomspace(freq[i]-bw[i]/2, freq[i]+bw[i]/2, Ngrid)
    freq = freq_grids

#signal

if use_nmt == False:
    DLdc = np.load(path+"power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy')

#foreground

mapfg = sim.get_fg_QU(freq, nside, dusttype=dusttype, synctype=synctype)

#get fg spectra
b_unbined=  nmt.NmtBin.from_lmax_linear(lmax=nside*3-1,nlb=1)

wspE_unbined = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b_unbined,purify='EE')
wspE = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b,purify='EE')
wspB_unbined = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b_unbined,purify='BB')
wspB = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b,purify='BB')
wspT = wspE # not needed here

ell_unbined= np.arange(3*nside)

CL_fg_EE = sim.computecross(mapfg,mapfg,mapfg,mapfg,wsp=wspE_unbined,Nell=len(ell_unbined),mask=mask,b=b_unbined,coupled=True,mode='EE')
CL_fg_BB = sim.computecross(mapfg,mapfg,mapfg,mapfg,wsp=wspB_unbined,Nell=len(ell_unbined),mask=mask,b=b_unbined,coupled=True,mode='BB')

#get noise spectra

CL_cross_noise_EE = np.zeros((Ncross,3*nside))
CL_cross_noise_BB = np.zeros((Ncross,3*nside))

if cl_noise == 'white':
	for i in range(Nfreqs):
		cross = cvl.cross_index(i, i, N_freqs)
		CL_cross_noise_EE[cross] = 4*np.pi/Npix * sigpix[i]**2
		CL_cross_noise_BB[cross] = Nls_EE[cross]
        
else:
    if masking_strat == '':
        Cl_noise = np.load(cl_noise+'CL_noise_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))[:, :, 1:, 2:2*nside]
    elif masking_strat == 'GWD':
        Cl_noise = np.load(cl_noise+'CL_noise_GWD_fsky%s_nside%s_aposcale%s.npy' % (fsky, nside, scale))[:, :, 1:, 2:2*nside]
    else:
        Cl_noise = np.load(cl_noise+'CL_noise_%s_%s_nside%s_aposcale%s.npy' % (masking_strat, complexity, nside, scale))[:, :, 1:, 2:2*nside]
    Cl_noise_mean = np.mean(Cl_noise, axis=(0,2))
    Cl_noise_std = np.std(Cl_noise, axis=(0,2))

    for i in range(N_freqs):
        cross = cvl.cross_index(i, i, N_freqs)
        p = [{'value': 4*np.pi/Npix * sigpix[i]**2, 'fixed': 0, 'limited': [1,1], 'limits': [0,np.inf]}]
        fa = {'Cl_noise': Cl_noise_mean[i], 'sigma_Cl': Cl_noise_std[i]}
        m = mpfit(fit.chi2_Nl, parinfo=p, functkw=fa, quiet=1)
        CL_cross_noise_EE[cross, 2:] = m.params[0] * np.ones(3*nside-2)
        CL_cross_noise_BB[cross] = CL_cross_noise_EE[cross]
        print(m.params[0], 4*np.pi/Npix * sigpix[i]**2)
raise ValueError
Nls_EE=[]
Nls_BB=[]
for i in range(N_freqs): 
    cross = cvl.cross_index(i, i, N_freqs)
    CL_cross_noise_EE[cross] /= Bls_EE[i]**2
    CL_cross_noise_BB[cross] /= Bls_BB[i]**2
    coupled_noise_EE = wspE_unbined.couple_cell([CL_cross_noise_EE[cross], np.zeros_like(CL_cross_noise_EE[cross]), np.zeros_like(CL_cross_noise_EE[cross]), CL_cross_noise_EE[cross]])[0]
    coupled_noise_BB = wspB_unbined.couple_cell([CL_cross_noise_BB[cross], np.zeros_like(CL_cross_noise_BB[cross]), np.zeros_like(CL_cross_noise_BB[cross]), CL_cross_noise_BB[cross]])[3]
    Nls_EE.append(coupled_noise_EE)
    Nls_BB.append(coupled_noise_BB)
Nls_EE=np.array(Nls_EE)
Nls_BB=np.array(Nls_BB)

#get cmb spectra
if cmb_e2e == False:
    CLcmb_or = hp.read_cl(path+'power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE
else:
    CLcmb_or = hp.read_cl(path+'power_spectra/Cls_LiteBIRD_e2e_r0.fits') #TT EE BB
CL_lens_EE = CLcmb_or[1,:nside*3]
CL_lens_BB = CLcmb_or[2,:nside*3]

coupled_cmb_EE = wspE_unbined.couple_cell([CL_lens_EE, np.zeros_like(CL_lens_EE), np.zeros_like(CL_lens_EE), CL_lens_BB])[0]
coupled_cmb_BB = wspB_unbined.couple_cell([CL_lens_EE, np.zeros_like(CL_lens_EE), np.zeros_like(CL_lens_EE), CL_lens_BB])[3]
CL_cmb_EE = np.array([ coupled_cmb_EE for i in range(N_freqs) for j in range(i, N_freqs)]) 
CL_cmb_BB = np.array([ coupled_cmb_BB for i in range(N_freqs) for j in range(i, N_freqs)]) 

if use_nmt==False:
    cov_sg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_signal=[np.zeros_like(DLdc[0,:,:Nell]), np.zeros_like(DLdc[0,:,:Nell]), DLdc[0,:,:Nell]], type='Knox_signal', output=mode_cov, progress=True)
    cov_an = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Knox-fg', output=mode_cov, progress=True)
    cov_anfg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Knox+fg', output=mode_cov, progress=True)

if use_nmt==True:
    cov_an   = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Nmt-fg', output=mode_cov, progress=True)
    #cov_anfg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Nmt+fg', output=mode_cov, progress=True)

if cmb_e2e:
    if dusttype == 1 and synctype == 1:
        dusttype, synctype = 'b', 'b'
    if dusttype == 10 and synctype == 5:
        dusttype, synctype = 'm', 'm'
    if dusttype == 12 and synctype == 7:
        dusttype, synctype = 'h', 'h'
    
if use_nmt==False:
    if masking_strat not in ['intersection', 'union']:
        np.save(path+'covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
        np.save(path+'covariances/cov_signal_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_sg)
        np.save(path+'covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
    else:
        np.save(path+'covariances/cov_Knox-fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
        np.save(path+'covariances/cov_signal_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_sg)
        np.save(path+'covariances/cov_Knox+fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)

if use_nmt==True:
    if masking_strat not in ['intersection', 'union']:
        np.save(path+'covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
        #np.save(path+'covariances/cov_Nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc'%(nside,fsky,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
    else:
        np.save(path+'covariances/cov_Nmt-fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_an)
        #np.save(path+'covariances/cov_Nmt+fg_nside%s_%s_scale%s_Nlbin%s_d%ss%sc'%(nside,masking_strat,scale,Nlbin,dusttype,synctype)+kw+'.npy',cov_anfg)
