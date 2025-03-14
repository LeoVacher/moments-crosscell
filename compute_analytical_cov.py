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
import pysm3
import time
from mpfit import mpfit
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import simu_lib as sim
import pysm3.units as u
import covlib as cvl 

r=0.
nside = 64
lmax = nside*2-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
synctype = 0
kw=''
use_nmt=True
mode_cov='BB'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()
fact_Dl= leff*(leff+1)/2/np.pi
Nell = len(leff)
instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
sens_P = instr['sens_P']
freq = instr['frequencies']
N_freqs=len(freq)
Ncross= int(N_freqs*(N_freqs+1)/2)
Npix = hp.nside2npix(nside)
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#signal

DLdc = np.load("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

#foreground

if dusttype==None and synctype==None:
    mapfg=np.zeros((N_freqs,2,Npix))
else:
    if dusttype==None:
        sky = pysm3.Sky(nside=512, preset_strings=['s%s'%synctype])#,'s%s'%synctype])
    if synctype==None:
        sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
    if synctype!=None and dusttype!=None:
        sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%synctype])
mapfg= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(N_freqs)])
mapfg=mapfg[:,1:]

#get fg spectra
b_unbined=  nmt.bins.NmtBin(nside=nside,lmax=nside*3-1,nlb=1)
wsp_unbined = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b_unbined)
wsp = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b)
ell_unbined= np.arange(3*nside)
fact_Dl_ub = ell_unbined*(ell_unbined+1)/2/np.pi

DLcross_fg = sim.computecross(mapfg,mapfg,mapfg,mapfg,wsp=wsp_unbined,mask=mask,fact_Dl=fact_Dl_ub,coupled=True,mode='all')
DL_fg_EE = DLcross_fg[0]
DL_fg_BB = DLcross_fg[3]
  
#get noise spectra
DL_cross_noise = np.ones((Ncross,3*nside))
z=0
Nls_EE=[]
Nls_BB=[]
for i in range(0,N_freqs): 
    for j in range(i,N_freqs): 
        DL_cross_noise[z]= fact_Dl_ub*4*np.pi*sigpix[i]*sigpix[j]/Npix
        coupled_noise = wsp_unbined.couple_cell([DL_cross_noise[z], np.zeros_like(DL_cross_noise[z]), np.zeros_like(DL_cross_noise[z]), DL_cross_noise[z]])
        Nls_EE.append(coupled_noise[0])
        Nls_BB.append(coupled_noise[3])
        z=z+1
Nls_EE=np.array(Nls_EE)
Nls_BB=np.array(Nls_BB)

#get cmb spectra
CLcmb_or=hp.read_cl('./power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE
DL_lens_EE = fact_Dl_ub*CLcmb_or[1,:len(fact_Dl_ub)]
DL_lens_BB = fact_Dl_ub*CLcmb_or[2,:len(fact_Dl_ub)]
DL_lens_EE=DL_lens_EE[:len(ell_unbined)]
DL_lens_BB=DL_lens_BB[:len(ell_unbined)]
coupled_cmb=wsp_unbined.couple_cell([DL_lens_EE, np.zeros_like(DL_lens_EE), np.zeros_like(DL_lens_EE), DL_lens_BB])
DL_cmb_EE = np.array([coupled_cmb[0] for i in range(N_freqs) for j in range(i, N_freqs)]) 
DL_cmb_BB = np.array([coupled_cmb[3] for i in range(N_freqs) for j in range(i, N_freqs)]) 

fsky_eff = np.mean(mask**2)

if use_nmt==False:
    cov_sg = cvl.compute_covmat(mask, wsp, Cls_signal_EE=None, Cls_signal_BB=DLdc[0,:,:Nell], Cls_cmb_EE=None, Cls_cmb_BB=None, Cls_fg_EE=None, Cls_fg_BB=None, Nls_EE=None, Nls_BB=None, type='Knox_signal', output=mode_cov, progress=True)
    cov_an = cvl.compute_covmat(mask, wsp, Cls_signal_EE=None, Cls_signal_BB=None, Cls_cmb_EE=None, Cls_cmb_BB=DL_cmb_BB, Cls_fg_EE=None, Cls_fg_BB=DL_fg_BB, Nls_EE=None, Nls_BB=Nls_BB, type='Knox-fg', output=mode_cov, progress=True)
    cov_anfg = cvl.compute_covmat(mask, wsp, Cls_signal_EE=None, Cls_signal_BB=None, Cls_cmb_EE=None, Cls_cmb_BB=DL_cmb_BB, Cls_fg_EE=None, Cls_fg_BB=DL_fg_BB, Nls_EE=None, Nls_BB=Nls_BB, type='Knox+fg', output=mode_cov, progress=True)

if use_nmt==True:
    cov_an   = cvl.compute_covmat(mask, wsp, Cls_signal_EE=None, Cls_signal_BB=None, Cls_cmb_EE=DL_cmb_EE/fsky_eff, Cls_cmb_BB=DL_cmb_BB/fsky_eff, Cls_fg_EE=DL_fg_EE/fsky_eff, Cls_fg_BB=DL_fg_BB/fsky_eff, Nls_EE=Nls_EE/fsky_eff, Nls_BB=Nls_BB/fsky_eff, type='Nmt-fg', output=mode_cov, progress=True)
    cov_anfg = cvl.compute_covmat(mask, wsp, Cls_signal_EE=None, Cls_signal_BB=None, Cls_cmb_EE=DL_cmb_EE/fsky_eff, Cls_cmb_BB=DL_cmb_BB/fsky_eff, Cls_fg_EE=DL_fg_EE/fsky_eff, Cls_fg_BB=DL_fg_BB/fsky_eff, Nls_EE=Nls_EE/fsky_eff, Nls_BB=Nls_BB/fsky_eff, type='Nmt+fg', output=mode_cov, progress=True)

if use_nmt==False:
    np.save('./covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_an)
    np.save('./covariances/cov_signal_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_sg)
    np.save('./covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_anfg)

if use_nmt==True:
    np.save('./covariances/cov_Nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_an)
    np.save('./covariances/cov_Nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_anfg)
