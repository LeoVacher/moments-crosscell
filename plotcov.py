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
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 15
dusttype = 0
synctype = 0
kw=''
use_nmt=True

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()
leff = leff[:ELLBOUND]
Nell = len(leff)
instr_name='litebird_full'
mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#signal

DLdc = np.load("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

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


if nmt==False:
    cov_sg =cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],sky=sky,instr_name=instr_name,type='signal',mask=mask,Linv=False,use_nmt=use_nmt,nside=nside,Nlbin=10,mode_cov='all')
cov_an = cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],sky=sky,instr_name=instr_name,type='Knox-fg',mask=mask,Linv=False,use_nmt=use_nmt,nside=nside,Nlbin=10,mode_cov='all')
cov_anfg = cvl.compute_analytical_cov(DL_signal=DLdc[:,:,:ELLBOUND],sky=sky,instr_name=instr_name,type='Knox+fg',mask=mask,Linv=False,use_nmt=use_nmt,nside=nside,Nlbin=10,mode_cov='all')

np.save('./covariances/invcov_nmt-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_an)
np.save('./covariances/invcov_nmt+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,synctype),cov_anfg)


binell=0

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
data_list = [np.log10(abs(cov_sim[binell])),np.log10(abs(cov_an[binell])), np.log10(abs(cov_sg[binell])), np.log10(abs(cov_anfg[binell]))]
titles = ["Sims", "Knox-fg", "Signal", "Knox+fg"]
for ax, data, title in zip(axes.flat, data_list, titles):
    im = ax.imshow(data,vmax=-2,vmin=-6)
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
plt.savefig('./cov_full.pdf')
plt.show()