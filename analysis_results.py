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
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
synctype = 0
fix=0
kw=''
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

#res1=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix0.npy'%(0,0),allow_pickle=True).item()
res1=np.load('Best-fits/results_d%ss%s_0.7_ds_o0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()
res2=np.load('Best-fits/results_d%ss%s_0.7_Nmt-fg_ds_o0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()
res3=np.load('Best-fits/results_d%ss%s_0.7_Nmt+fg_ds_0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()
res4=np.load('Best-fits/results_d%ss%s_0.7_signal_ds_0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()

legs1= 'sims'
legs2= 'Knox-fg'
legs3= 'Knox+fg'
legs4= 'signal'

c1='darkblue'
c2='darkorange'
c3='forestgreen'
c4='darkred'

mom_an=np.load('./analytical_mom/analytical_mom_nside64_fsky0.7_scale10_Nlbin10_d%ss%s.npy'%(dusttype,synctype),allow_pickle=True).item()

reslist= [res1]
leglist= [legs1]
collist= [c1]

plotrespdf(l,reslist,leglist,collist,mom_an)