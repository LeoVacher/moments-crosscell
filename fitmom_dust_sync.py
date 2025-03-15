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
order_to_fit= ['0','1bt'] 
Pathload = './'
all_ell = False #all ell or each ell independently
fix = 1 #fix beta and T ?
adaptative = False
N = 500
parallel = False
cov_type = 'Nmt-fg' #choices: sim, Knox-fg, Knox+fg, Nmt-fg, Nmt+fg, signal.
kw=''
dusttype_cov = 0
synctype_cov = 0

if cov_type != 'sim':
    kw += '_%s'%cov_type

if parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

# Call C_ell of simulation

if synctype == None:
    DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy"%(nside,fsky,scale,Nlbin,dusttype))
else:
    DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
Nell = len(l)

#instrument informations:

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
freq = freq
nf = len(freq)
Ncross = int(nf*(nf+1)/2)

#compute cross-frequencies 

nucross = []
for i in range(0,nf):
    for j in range(i,nf):
        nucross.append(np.sqrt(freq[i]*freq[j]))
nucross = np.array(nucross)

#compute Cholesky matrix:

if np.shape(np.argwhere(DLdc == 0))[0] == 0:
    Ncov = len(DLdc)
else:
    Ncov = np.argwhere(DLdc == 0)[0,0]-1

if all_ell:
    if cov_type == 'sim':
        Linvdc = cvl.getLinv_all_ell(DLdc[:Ncov,:,:Nell],printdiag=True)
    else:
        cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_all_ell.npy"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov))
        Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=False, return_cholesky=True, return_new=False)

else:
    if cov_type == 'sim':
        Linvdc = cvl.getLinvdiag(DLdc[:Ncov,:,:Nell],printdiag=True)
    else:
        cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov))
        Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=True, return_cholesky=True, return_new=False)

#N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.
DLdc = DLdc[:N,:,:Nell]

# fit MBB and PL, get results, save and plot

if '0' in order_to_fit:
    p0 = [100, 1.54, 20, 10, -3,0, 0] #first guess for mbb A, beta, T, A_s, beta_s, A_sd and r
    results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw)

# fit order 1 in beta and T, get results, save and plot

if '1bt' in order_to_fit:
    p0 = [100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
    results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix,all_ell=all_ell,adaptative=adaptative,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw)

# fit order 1 in beta, T and beta_s, get results, save and plot

if '1bts' in order_to_fit:
    p0 = [100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
    results_ds_o1bts = an.fit_mom('ds_o1bts',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw)

