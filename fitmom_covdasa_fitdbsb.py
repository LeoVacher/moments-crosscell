import sys
sys.path.append("./lib")
import numpy as np
import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import mpfitlib as mpl
import scipy
#from Nearest_Positive_Definite import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
from plotlib import plotr_gaussproduct
from plotlib import plotrespdf
import matplotlib.pyplot as plt 
import covlib as cvl

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.8
ELLBOUND = 15
dusttype_fit = 'b'
synctype_fit = 'b'
dusttype_cov = 1
synctype_cov = 1
all_ell=False 
fix=0
kw='_covd%ss%s'%(dusttype_cov,synctype_cov)
kwsim=''
Pathload='./'

# Call C_ell of simulation

if synctype_fit==None:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy"%(nside,fsky,scale,Nlbin,dusttype_fit))
else:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype_fit,synctype_fit))

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

#instrument informations:

instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
freq=freq
nf = len(freq)
Ncross = int(nf*(nf+1)/2)

#compute cross-frequencies 

nucross = []
for i in range(0,nf):
    for j in range(i,nf):
        nucross.append(np.sqrt(freq[i]*freq[j]))
nucross = np.array(nucross)

N = 249#len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.

DLdc=DLdc[:N,:,:Nell]

#compute Cholesky matrix:

DL_cov = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov))
if np.shape(np.argwhere(DL_cov == 0))[0] == 0:
    Ncov=len(DL_cov)
else:
    Ncov=np.argwhere(DL_cov == 0)[0,0]-1
DL_cov=DL_cov[:Ncov]
if all_ell==True:
    Linvdc=cvl.getLinv_all_ell(DL_cov,printdiag=True)
else:
    Linvdc=cvl.getLinvdiag(DL_cov,printdiag=True)

# fit MBB and PL, get results, save and plot

p0=[100, 1.50, 20, 10, -3,0, 0] #first guess for mbb A, beta, T, A_s, beta_s, A_sd and r
results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype_fit,synctype_fit,fsky)+kw)

# fit order 1 in beta and T, get results, save and plot

p0=[100, 1.50, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype_fit,synctype_fit,fsky)+kw)

# fit order 1 in beta, T and beta_s, get results, save and plot

p0=[100, 1.50, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
results_ds_o1bts = an.fit_mom('ds_o1bts',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype_fit,synctype_fit,fsky)+kw)
