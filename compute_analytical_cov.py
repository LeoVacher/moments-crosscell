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
import mpfitlib as mpl
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import simu_lib as sim
import pysm3.units as u
import covlib as cvl 

def computecross(map_FM1,map_FM2,map_HM1,map_HM2):
    N_freqs=len(map_HM1)
    Ncross=int(N_freqs*(N_freqs+1)/2)
    CLcross=np.zeros((Ncross,19))
    z=0
    for i in range(0,N_freqs):
        for j in range(i,N_freqs):
            if i != j :
                CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_FM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_FM2[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            if i==j :
                CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*map_HM1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*map_HM2[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            z = z +1
    return fact_Dl[:15]*CLcross[:,:15]

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 15
dusttype = 1
synctype = 1
kw=''
 
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()
leff = leff[:ELLBOUND]
Nell = len(leff)
instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
sens_P= instr['sens_P']
Npix = hp.nside2npix(nside)
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))
N_freqs=len(freq)
fact_Dl= leff*(leff+1)/2/np.pi
nf = len(freq)
Ncross = int(nf*(nf+1)/2)

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#signal

DLdc = np.load("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

#cmb spectra:

CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE
DL_lens = fact_Dl*b.bin_cell(CLcmb_or[2,:lmax+1])[:15]
DL_cross_lens = np.array([DL_lens for i in range(N_freqs) for j in range(i, N_freqs)])

#foreground spectra:

if dusttype==None and synctype==None:
    mapfg=np.zeros((N_freqs,2,Npix))
else:
    if dusttype==None:
        sky = pysm3.Sky(nside=512, preset_strings=['s%s'%synctype])#,'s%s'%synctype])
    if synctype==None:
        sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
    if synctype!=None and dusttype!=None:
        sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%synctype])
    mapfg= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
    mapfg=mapfg[:,1:]

wsp_dc=[]
for i in range(0,N_freqs): 
    for j in range(i,N_freqs):
        w_dc = nmt.NmtWorkspace()
        if i != j :
            w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask,1*mapfg[j],purify_e=False, purify_b=True), b)
        if i==j :
            w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapfg[j],purify_e=False, purify_b=True), b)
        wsp_dc.append(w_dc)
 
wsp_dc=np.array(wsp_dc)

DLcross_fg = computecross(mapfg,mapfg,mapfg,mapfg)

# noise spectra:

DL_cross_noise=np.ones((Ncross,len(leff)))
z=0
for i in range(0,N_freqs): 
    for j in range(i,N_freqs):
        DL_cross_noise[z]= fact_Dl*4*np.pi*sigpix[i]*sigpix[j]/Npix*DL_cross_noise[z]
        z=z+1

#compute and save Cholesky matrix of the inverse covariance using appropriate functions:

Linv_an= cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],DLcross_fg=DLcross_fg,DL_cross_lens=DL_cross_lens,DL_cross_noise=DL_cross_noise,type='Knox-fg',ell=leff,Nlbin=10,mask=mask,Linv=True)
#Linv_sg=cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],DLcross_fg=DLcross_fg,DL_cross_lens=DL_cross_lens,DL_cross_noise=DL_cross_noise,type='signal',ell=leff,Nlbin=10,mask=mask,Linv=True)
Linv_anfg= cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],DLcross_fg=DLcross_fg,DL_cross_lens=DL_cross_lens,DL_cross_noise=DL_cross_noise,type='Knox+fg',ell=leff,Nlbin=10,mask=mask,Linv=True)

np.save('./covariances/Linv_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),Linv_an)
np.save('./covariances/Linv_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),Linv_anfg)
