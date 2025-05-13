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
from plotlib import plotrespdf
import analytical_mom_lib as anmomlib
import healpy as hp

r=0.
nside = 64
lmax = nside*3-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 10
synctype = 5
Pathload = './'
nu0d = 402.
nu0s = 40.
load = False
adaptative = False
N = 50
cov_type = 'sim' #choices: sim, Knox-fg, Knox+fg, Nmt-fg, Nmt+fg, signal.
kw=''
dusttype_cov = dusttype
synctype_cov = synctype
pivot_o0 = True
iterate = False
fixr=1
all_ell_o0 = False

if cov_type != 'sim':
    kw += '_%s'%cov_type
if iterate == True :
    kw += '_iterate'
if pivot_o0:
    kw += '_pivoto0'
if nu0d != 353.:
    kw += '_nu0d%s'%nu0d
if nu0s != 23.:
    kw += '_nu0s%s'%nu0s

# Call C_ell of simulation

# if synctype == None:
#     DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy"%(nside,fsky,scale,Nlbin,dusttype))
# else:
#     DLdc = np.load(Pathload+"/power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

DLdc = np.array([np.load('/global/u1/l/leovchr/codes/moments-crosscell/fit-Samy/LB_d10s5_nside64_delta10_planck70apo10/%s.npy'%i) for i in range(500)])
DLdc= DLdc[:,2]

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
l = b.get_effective_ells()
Nell = 12#len(l)

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

#if cov_type == 'sim':
#    if all_ell_o0:
#        Linvdc = cvl.getLinv_all_ell(DLdc[:Ncov,:,:Nell],printdiag=True)
#    else:
 #       Linvdc = cvl.getLinvdiag(DLdc[:Ncov,:,:Nell],printdiag=True)
#else:
#    cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc_all_ell.npy"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov))
#    Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=False, return_cholesky=True, return_new=False)

cov = np.load('/global/u1/l/leovchr/codes/moments-crosscell/fit-Samy/LB_d10s5_nside64_delta10_planck70apo10/covmat_Nmt-fg_all_uK_CMB.npy')
Linvdc = cvl.inverse_covmat(cov[2], Ncross, neglect_corbins=False, return_cholesky=True, return_new=False)
#N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.
DLdc = DLdc[:N,:,:Nell]

#first guesses:

betabar, tempbar, betasbar = 1.5, 20, -3

if pivot_o0:
    p0 = [abs(DLdc[0,-1]), betabar, tempbar, abs(DLdc[0,0]), betasbar,0, 0] #first guess for mbb A, beta, T, A_s, beta_s, A_sd and r
    if load:
        try:
            results_ds_o0 = np.load('best_fits/results_d%ss%s_%s_ds_o%s_fix%s_all_ell.npy'%(dusttype,synctype,fsky,'0','0'),allow_pickle=True).item()
        except:
            results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=0, all_ell=all_ell_o0,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=False,nu0d=nu0d,nu0s=nu0s,fixr=fixr)
    else:
        results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=0, all_ell=all_ell_o0,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=False,nu0d=nu0d,nu0s=nu0s,fixr=fixr)

    #update with order 0's best fit:

    betabar = np.mean(results_ds_o0['beta_d'])
    tempbar = np.mean(results_ds_o0['T_d'])
    betasbar = np.mean(results_ds_o0['beta_s'])

    if fsky==1:
        mask = np.ones(hp.nside2npix(nside))
    else:
        mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

    try:
        mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar), allow_pickle=True).item()
        print('mom found')
    except:
        mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s)
        np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar,nu0d,nu0s), mom_an)

    reslist = [results_ds_o0]
    leglist = ['d%ss%s_o0_fsky%s_full%s'%(dusttype,synctype,fsky,kw)]
    collist = ['darkred']
    plotrespdf(l[:12],reslist,leglist,collist,mom_an,plot_contours=False,betadbar=betabar,tempbar=tempbar,betasbar=betasbar)

# fit order 1 in beta and T, get results, save and plot

if cov_type == 'sim':
    Linvdc = cvl.getLinvdiag(DLdc[:Ncov,:,:Nell],printdiag=True)
else:
    cov = np.load(Pathload+"/covariances/cov_%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(cov_type,nside,fsky,scale,Nlbin,dusttype_cov,synctype_cov))
    Linvdc = cvl.inverse_covmat(cov, Ncross, neglect_corbins=True, return_cholesky=True, return_new=False)

p0 = [100, betabar, tempbar, 10, betasbar, 1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]

if load:
    try:
        results_ds_o1bt = np.load('best_fits/results_d%ss%s_%s_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,'1bt','1'),allow_pickle=True).item()
    except:
        results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=1,all_ell=False,adaptative=adaptative,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=False,iterate=iterate,nu0d=nu0d,nu0s=nu0s,fixr=fixr)
else:
    results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin, fix=1, all_ell=False,adaptative=adaptative,kwsave='d%ss%s_%s'%(dusttype,synctype,fsky)+kw,plotres=False,iterate=iterate,nu0d=nu0d,nu0s=nu0s,fixr=fixr)

if fsky==1:
    mask = np.ones(hp.nside2npix(nside))
else:
    mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

try:
    mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar), allow_pickle=True).item()
except:
    mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s)
    np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, betabar, tempbar, betasbar,nu0d,nu0s), mom_an)

reslist = [results_ds_o1bt]
leglist = ['d%ss%s_o1bt_fsky%s_full%s'%(dusttype,synctype,fsky,kw)]
collist = ['darkred']

plotrespdf(l[:12],reslist,leglist,collist,mom_an,plot_contours=True,betadbar=betabar,tempbar=tempbar,betasbar=betasbar)
