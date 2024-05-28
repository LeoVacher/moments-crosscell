import sys
sys.path.append("./lib")

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
from analys_lib import plotr_gaussproduct, plotmed
from matplotlib.backends.backend_pdf import PdfPages 
from plotlib import plotrespdf
from matplotlib.backends.backend_pdf import PdfPages 

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
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

#res1=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix0.npy'%(0,0),allow_pickle=True).item()
res1=np.load('Best-fits/resultso1bt_PL_d%ss%sc_fix0.npy'%(0,0),allow_pickle=True).item()
res2=np.load('Best-fits/resultsmbb_PL_d%ss%sc.npy'%(0,0),allow_pickle=True).item()

legs1='d0s0'
legs2='d1s1'

c1='darkblue'
c2='darkorange'

plotrespdf(l,[res1,res2],[legs1,legs2],[c1,c2])