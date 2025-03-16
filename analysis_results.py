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

r=0.
nside = 64
lmax = nside*2-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
synctype = 0
fix=0
order = '1bt' #0 or 1bt
kw = ''
kwsim = ''
Pathload = './'
plot_contours=True

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
Nell = len(l)

res1 = np.load('best_fits/results_d%ss%s_%s_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
res2 = np.load('best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
res3 = np.load('best_fits/results_d%ss%s_%s_Nmt+fg_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()
res4 = np.load('best_fits/results_d%ss%s_%s_signal_ds_o%s_fix%s.npy'%(dusttype,synctype,fsky,order,fix),allow_pickle=True).item()

legs1 = 'sims'
legs2 = 'Knox-fg'
legs3 = 'Knox+fg'
legs4 = 'signal'

c1 = 'darkblue'
c2 = 'darkorange'
c3 = 'forestgreen'
c4 = 'darkred'

mom_an = np.load('./analytical_mom/analytical_mom_nside%s_fsky%s_scale10_Nlbin10_d%ss%s.npy'%(nside,fsky,dusttype,synctype),allow_pickle=True).item()

reslist = [res1,res2,res3]
leglist = [legs1,legs2,legs3]
collist = [c1,c2,c3]

plotrespdf(l,reslist,leglist,collist,mom_an,plot_contours=plot_contours)