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

#### Pipeline to fit different models on cross-frequency power spectra from simulations ####

nside = 64 #HEALPix nside
lmax = nside*3-1 #maximum multipole
scale = 10 #scale of apodisaton of the mask
Nlbin = 10 #binning for bandpower
fsky = 0.8 #sky fraction of the raw mask
dusttype = 1 #index of Pysm's dust model
synctype = 1 #index of Pysm's synchrotron model
order_to_fit= ['1bts'] #orders to fit ('0', '1bt' or '1bts')
Pathload = './' #Home path
all_ell = False #all ell or each ell independently (True/False)
fix = 1 #fix beta and T (0:fit, 1:fix)?
fixr= 0 #fix r (0:fit, 1:fix)?
adaptative = True #adapt to fix to 0 non detected moments (True/False)
N = 250 #number of simulations
plotres = True #plot and save pdf?
parallel = False #parallelize?
pivot_o0 = False #use the best fit of order 0?
iterate = False #iterate to obtain ideal ell-dependent pivots (True/False)
cov_type = 'Nmt-fg' #choices: sim, Knox-fg, Knox+fg, Nmt-fg, Nmt+fg, signal.
kw='' #additional keyword to add?
kws='' #keyword for the simulations?
dusttype_cov = dusttype #dust type for the covariance matrix
synctype_cov = synctype #synchrotron type for the covariance matrix
nu0d = 402. #dust reference frequency
nu0s = 40. #synchrotron reference frequency
gaussbeam = False #are simulations smoothed with gaussian beam?


if cov_type != 'sim':
    kw += '_%s'%cov_type
if iterate == True :
    kw+= '_iterate'
if fixr==1:
    kw+= '_fixr'
if gaussbeam:
    kws += '_gaussbeam'

kw += kws

if parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

# Call C_ell of simulation

if synctype == None:
    DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(nside,fsky,scale,Nlbin,dusttype)+kws+'.npy')
else:
    DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,synctype)+kws+'.npy')

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
l = b.get_effective_ells()
Nell = 12#len(l)

#instrument informations:

instr_name = 'litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
#freq = np.sort(freq)
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
        cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov)+kws+'.npy')[:Ncross*Nell, :Ncross*Nell]
        Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=False, return_cholesky=True, return_new=False)

else:
    if cov_type == 'sim':
        Linvdc = cvl.getLinvdiag(DLdc[:Ncov,:,:Nell],printdiag=True)
    else:
        cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov)+kws+'.npy')[:Ncross*Nell, :Ncross*Nell]
        Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=True, return_cholesky=True, return_new=False)

#N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.
DLdc = DLdc[:N,:,:Nell]

if pivot_o0:
    try:
        o0 = np.load('best_fits/results_d%ss%s_%s_%s_ds_o%s_fix%s_all_ell.npy'%(dusttype,synctype,fsky,cov_type,'0','0'),allow_pickle=True).item()
    except:
        if cov_type == 'sim':
            Linvdc0 = cvl.getLinv_all_ell(DLdc[:Ncov,:,:Nell],printdiag=True)
        else:
            Linvdc0 = cvl.inverse_covmat(cov, Ncross, neglect_corbins=False, return_cholesky=True, return_new=False)
        p0 = [np.abs(DLdc[0,-1]), 1.5, 20, np.abs(DLdc[0,0]), -3,0, 0] #first guess for mbb A, beta, T, A_s, beta_s, A_sd and r
        o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc0,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=0, all_ell=True,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=False,iterate=False,nu0d=nu0d,nu0s=nu0s,fixr=fixr)
    betabar = np.mean(o0['beta_d'])
    tempbar = np.mean(o0['T_d'])
    betasbar = np.mean(o0['beta_s'])
else:
    betabar, tempbar, betasbar = 1.5, 20, -3

# fit MBB and PL, get results, save and plot

if '0' in order_to_fit:
    p0 = [np.abs(DLdc[0,-1]), betabar, tempbar, np.abs(DLdc[0,0]), betasbar,0, 0] #first guess for mbb A, beta, T, A_s, beta_s, A_sd and r
    results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=plotres,iterate=iterate,nu0d=nu0d,nu0s=nu0s,fixr=fixr)

# fit order 1 in beta and T, get results, save and plot

if '1bt' in order_to_fit:
    p0 = [np.abs(DLdc[0,-1]), betabar, tempbar, np.abs(DLdc[0,0]), betasbar,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
    results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix,all_ell=all_ell,adaptative=adaptative,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=plotres,iterate=iterate,nu0d=nu0d,nu0s=nu0s,fixr=fixr)

# fit order 1 in beta, T and beta_s, get results, save and plot

if '1bts' in order_to_fit:
    p0 = [np.abs(DLdc[0,-1]), betabar, tempbar, np.abs(DLdc[0,0]), betasbar,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
    results_ds_o1bts = an.fit_mom('ds_o1bts',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=fix, all_ell=all_ell,adaptative=adaptative,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=plotres,iterate=iterate,nu0d=nu0d,nu0s=nu0s,fixr=fixr)


