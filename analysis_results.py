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

nres=2
nside = 64
lmax = nside*2-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 10
synctype = 5
fix=1
order = '1bt' #0 or 1bt
kw = ''
kwsim = ''
Pathload = './'
plot_contours=False
all_ell=False
nu0d=402.
nu0s=40.

if all_ell==True:
    kw=kw+'_all_ell'

if fsky==1:
    mask = np.ones(hp.nside2npix(nside))
else:
    mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
l = b.get_effective_ells()
Nell = len(l)

res1 = np.load('best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
res2 = np.load('best_fits/results_d%ss%s_%s_Nmt-fg_fixr_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res3 = np.load('best_fits/results_d%ss%s_%s_Nmt+fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res4 = np.load('best_fits/results_d%ss%s_%s_signal_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()

legs1 = 'd%ss%s_fsky%s_full'%(dusttype,synctype,fsky)
legs2 = 'fixr'
legs3 = 'Knox+fg'
legs4 = 'signal'

c1 = 'darkblue'
c2 = 'darkorange'
c3 = 'forestgreen'
c4 = 'darkred'

betabar = np.mean(res1['beta_d'])
tempbar = np.mean(res1['T_d'])
betasbar= np.mean(res1['beta_s'])

try:
    mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), allow_pickle=True).item()
except:
    print('Computing theoretical expecations for the fitted quantities ...')
    mom_an = anmomlib.getmom(dusttype, synctype, betabar, tempbar, betasbar, mask, Nlbin=Nlbin, nside=nside,nu0d=nu0d,nu0s=nu0s)
    np.save('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s_%s%s%s_%s%s.npy' % (nside, fsky, dusttype, synctype, np.round(betabar,3), np.round(tempbar,3), np.round(betasbar,3),int(nu0d),int(nu0s)), mom_an)

reslist = [globals()[f"res{i}"] for i in range(1, nres + 1)]
leglist = [globals()[f"legs{i}"] for i in range(1, nres + 1)]
collist = [globals()[f"c{i}"] for i in range(1, nres + 1)]

plotrespdf(l,reslist,leglist,collist,mom_an,plot_contours=plot_contours,betadbar=betabar,tempbar=tempbar,betasbar=betasbar)
