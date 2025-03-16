import sys
sys.path.append("./lib")

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
from matplotlib.backends.backend_pdf import PdfPages 

nres=1
nside = 64
lmax = nside*2-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
synctype = 0
fix=1
order = '1bt' #0 or 1bt
kw = ''
kwsim = ''
Pathload = './'
plot_contours=False

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
Nell = len(l)

res1 = np.load('best_fits/results_d%ss%s_%s_ds_o%s_fix%s_all_ell.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res2 = np.load('best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res3 = np.load('best_fits/results_d%ss%s_%s_Nmt+fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
#res4 = np.load('best_fits/results_d%ss%s_%s_signal_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()

legs1 = 'sims'
legs2 = 'Knox-fg'
legs3 = 'Knox+fg'
legs4 = 'signal'

c1 = 'darkblue'
c2 = 'darkorange'
c3 = 'forestgreen'
c4 = 'darkred'

mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s.npy'%(nside,fsky,dusttype,synctype),allow_pickle=True).item()

reslist = [globals()[f"res{i}"] for i in range(1, nres + 1)]
leglist = [globals()[f"legs{i}"] for i in range(1, nres + 1)]
collist = [globals()[f"c{i}"] for i in range(1, nres + 1)]

plotrespdf(l,reslist,leglist,collist,mom_an,plot_contours=plot_contours)
