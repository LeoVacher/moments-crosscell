import sys
sys.path.append("./lib")

import healpy as hp
import numpy as np
import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import scipy
#from Nearest_Positive_Definite import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
from matplotlib.backends.backend_pdf import PdfPages 
from plotlib import plotrespdf
import analytical_mom_lib as anmomlib

nres=1 #number of results to plot
nside = 64 #HEALPix nside
lmax = nside*2-1 #maximum multipole
scale = 10 #scale of apodisaton of the mask
Nlbin = 10 #binning for bandpower
fsky = 0.7 #sky fraction of the raw mask
dusttype = 10 #index of Pysm's dust model
synctype = 5 #index of Pysm's synchrotron model
fix=1 #fix beta and T (0:fit, 1:fix)?
order = '1bts' #0, 1bt or 1bts 
kw = '' #keyword for the fitting scheme
kwsim = '' #keyword for the simulation 
Pathload = './' #Home path
plot_contours=True #plot contours for best fit parameters
all_ell=False #all ell or each ell independently (True/False)
nu0d=402. #dust reference frequency
nu0s=40. #synchrotron reference frequency
ell_pivot = True #plot theoretical ideal pivot as a function of ell
mode = 'EE' #mode to analyze. Should be 'EE' or 'BB'

if all_ell==True:
    kw=kw+'_all_ell'

if fsky==1:
    mask = np.ones(hp.nside2npix(nside))
else:
    mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
l = b.get_effective_ells()
Nell = len(l)

#loads the results:
res1 = np.load('best_fits/results_d%ss%s_%s_%s_Nmt-fg_iterate_fixr_gaussbeam_ds_o%s_fix%s_iterate.npy'%(dusttype,synctype,fsky,mode,order,fix),allow_pickle=True).item()
#res2 = np.load('best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s_adaptative.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res3 = np.load('best_fits/results_d%ss%s_%s_Nmt+fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res4 = np.load('best_fits/results_d%ss%s_%s_signal_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()

#labels and legend for the results
legs1 = 'd%ss%s_%s_fsky%s'%(dusttype,synctype,mode,fsky)
legs2 = 'adaptive'
legs3 = 'Knox+fg'
legs4 = 'signal'

#colors for the results
c1 = 'darkblue'
c2 = 'darkorange'
c3 = 'forestgreen'
c4 = 'darkred'

#Which pivots for the moments?
betabar = np.median(res1['beta_d'][~np.isnan(res1['X2red'])])
tempbar = np.median(res1['T_d'][~np.isnan(res1['X2red'])])
betasbar= np.median(res1['beta_s'][~np.isnan(res1['X2red'])])

#load analytical moments:
try:
    mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, mode, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), allow_pickle=True).item()
except:
    print('Computing theoretical expecations for the fitted quantities ...')
    mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s, mode=mode)
    np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, mode, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), mom_an)

#plot:

reslist = [globals()[f"res{i}"] for i in range(1, nres + 1)]
leglist = [globals()[f"legs{i}"] for i in range(1, nres + 1)]
collist = [globals()[f"c{i}"] for i in range(1, nres + 1)]

plotrespdf(l,reslist,leglist,collist,mom_an,plot_contours=plot_contours,betadbar=betabar,tempbar=tempbar,betasbar=betasbar,ell_pivot=ell_pivot)
