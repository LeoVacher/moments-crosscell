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
import analys_lib as an

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 19
dusttype = 0
synctype = None
kw=''
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

res=np.load('Best-fits/resultsPL_s0c.npy',allow_pickle=True).item()
an.plotmed(l,'X2red',res)
an.plotmed(l,'beta_s',res)
an.plotmed(l,'A_s',res)
an.plotmed(l,'r',res)

an.plotr_gaussproduct(res,Nmax=15,color='darkorange')

res2=np.load('Best-fits/resultso1bs_s0c.npy',allow_pickle=True).item()

an.plotmed(l,'X2red',res,show=False,legend='PL')
an.plotmed(l,'X2red',res2,show=False,legend='o1',color='darkorange')
plt.show()
an.plotmed(l,'beta_s',res2)
an.plotmed(l,'A_s',res2)
an.plotmed(l,'r',res2)
an.plotmed(l,'Asw1bs',res2)
an.plotmed(l,'w1bsw1bs',res2)

an.plotr_gaussproduct(res2,Nmax=15,color='darkorange')

