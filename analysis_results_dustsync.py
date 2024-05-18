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
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
from analys_lib import plotr_gaussproduct, plotmed

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 15
dusttype = 1
synctype = 1
kw=''
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)


res0=np.load('Best-fits/resultsmbb_PL_d1s1c.npy',allow_pickle=True).item()
res1=np.load('Best-fits/resultso1bt_PL_d1s1c_fix0.npy',allow_pickle=True).item()
res2=np.load('Best-fits/resultso1bt_PL_d1s1c_fix1.npy',allow_pickle=True).item()

legs0='PL'
legs1='fix0'
legs2='fix1'

res0['X2red'] = res0['X2red'][:,:50]
plotmed(l,'X2red',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+1,'X2red',res1,show=False,color='darkred',legend=legs1)
plotmed(l+2,'X2red',res2,show=False,legend=legs2)
plt.show()

res0['A_s'] = res0['A_s'][:,:50]
plotmed(l,'A_s',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+1,'A_s',res1,show=False,color='darkred',legend=legs1)
plotmed(l+2,'A_s',res2,show=False,legend=legs2)
plt.show()

res0['beta_s'] = res0['beta_s'][:,:50]
plotmed(l+1,'beta_s',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+2,'beta_s',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'beta_s',res2,show=False,legend=legs2)
plt.show()

res0['beta'] = res0['beta'][:,:50]
plotmed(l+1,'beta',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+2,'beta',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'beta',res2,show=False,legend=legs2)
plt.show()

res0['temp'] = res0['temp'][:,:50]
plotmed(l+1,'temp',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+2,'temp',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'temp',res2,show=False,legend=legs2)
plt.show()

res0['r'] = res0['r'][:,:50]
plotmed(l+1,'r',res0,show=False,color='darkorange',legend=legs0)
plotmed(l+2,'r',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'r',res2,show=False,legend=legs2)
plt.show()

plotr_gaussproduct(res0,Nmin=2,Nmax=15,debug=False,color='darkblue')
plotr_gaussproduct(res1,Nmin=2,Nmax=15,debug=False,color='darkred')
plotr_gaussproduct(res2,Nmin=2,Nmax=15,debug=False,color='darkorange')

plotmed(l+2,'Asw1b',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'Asw1b',res2,show=False,legend=legs2)
plt.show()

plotmed(l+2,'Asw1t',res1,show=False,color='darkred',legend=legs1)
plotmed(l,'Asw1t',res2,show=False,legend=legs2)
plt.show()