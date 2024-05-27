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
dusttype = 1
synctype = 0
kw=''
kwsim=''
Pathload='./'

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

resdust=np.load('Best-fits/resultso1bt_d1c.npy',allow_pickle=True).item()
resdustsync=np.load('Best-fits/resultso1bt_PL_d1s1c.npy',allow_pickle=True).item()
resdustsyncfull=np.load('Best-fits/resultso1bt_moms_full_d1s1c.npy',allow_pickle=True).item()


an.plotmed(l,'Aw1b',resdustsync,show=False,legend='d1s1+o1bt')
an.plotmed(l,'Aw1b',resdustsyncfull,color="darkorange",show=False,legend='d1s1+o1bts')
an.plotmed(l,'Aw1b',resdust,color="darkred",show=False,legend='d1+o1bt')
plt.show()

an.plotmed(l,'w1bsw1bs',resdustsyncfull,color="darkorange",legend='d1s1+o1bts')

an.plotr_gaussproduct(resdustsyncfull,Nmin=2,Nmax=15,debug=True,color='darkorange')


