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
import covlib as cvl 

r=0.
nside = 64
lmax = nside*3-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 9
synctype = 4
kw=''
use_nmt=True
mode_cov='BB'

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

Bls_EE = np.zeros((N_freqs, 3*nside))
Bls_BB = np.zeros((N_freqs, 3*nside))

for i in range(N_freqs):
    Bls_EE[i], Bls_BB[i] = hp.gauss_beam(beam[i], lmax=3*nside-1, pol=True).T[1:3]

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))
fsky_eff = np.mean(mask**2)

#signal

if use_nmt == False:
    DLdc = np.load("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

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
CL_cross_noise_EE = np.ones((Ncross,3*nside))
CL_cross_noise_BB = np.ones((Ncross,3*nside))
z=0
Nls_EE=[]
Nls_BB=[]
for i in range(0,N_freqs): 
    for j in range(i,N_freqs): 
        CL_cross_noise_EE[z]= 4*np.pi*sigpix[i]*sigpix[j]/Npix / (Bls_EE[i] * Bls_EE[j])
        CL_cross_noise_BB[z]= 4*np.pi*sigpix[i]*sigpix[j]/Npix / (Bls_BB[i] * Bls_BB[j])
        coupled_noise_EE = wspE_unbined.couple_cell([CL_cross_noise_EE[z], np.zeros_like(CL_cross_noise_EE[z]), np.zeros_like(CL_cross_noise_EE[z]), CL_cross_noise_EE[z]])[0]
        coupled_noise_BB = wspB_unbined.couple_cell([CL_cross_noise_BB[z], np.zeros_like(CL_cross_noise_BB[z]), np.zeros_like(CL_cross_noise_BB[z]), CL_cross_noise_BB[z]])[3]
        Nls_EE.append(coupled_noise_EE)
        Nls_BB.append(coupled_noise_BB)
        z=z+1
Nls_EE=np.array(Nls_EE)
Nls_BB=np.array(Nls_BB)

#get cmb spectra
CLcmb_or = hp.read_cl('./power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE
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
    cov_anfg = cvl.compute_covmat(mask, [wspT, wspE, wspB], Cls_cmb=[np.zeros_like(CL_cmb_EE), CL_cmb_EE/fsky_eff, CL_cmb_BB/fsky_eff], Cls_fg=[np.zeros_like(CL_fg_EE), CL_fg_EE/fsky_eff, CL_fg_BB/fsky_eff], Nls=[np.zeros_like(Nls_EE), Nls_EE/fsky_eff, Nls_BB/fsky_eff], type='Nmt+fg', output=mode_cov, progress=True)

if use_nmt==False:
    np.save('./covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_an)
    np.save('./covariances/cov_signal_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_sg)
    np.save('./covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_anfg)

if use_nmt==True:
    np.save('./covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_an)
    np.save('./covariances/cov_Nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_anfg)
