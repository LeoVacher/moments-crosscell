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
import datetime
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import analys_lib as an

dusttype=0
nside = 256
lmax = nside*3-1
Nlbin = 10
ELLBOUND = 20

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
ell = b.get_effective_ells()
ell = ell[:ELLBOUND]
Nell = len(ell)

#load dictionnaries:

resultsmbb =np.load('Best-fits/resultsmbb_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso1b =np.load('Best-fits/resultso1b_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso1bt =np.load('Best-fits/resultso1bt_d%sc.npy'%dusttype,allow_pickle=True).item()
resultso2b =np.load('Best-fits/resultso2b_d%sc.npy'%dusttype,allow_pickle=True).item()

plt.scatter(ell,np.median(resultsmbb['X2red'],axis=1),label="mbb")
plt.scatter(ell,np.median(resultso1b['X2red'],axis=1),label="o1b")
plt.scatter(ell,np.median(resultso1bt['X2red'],axis=1),label="o1bt")
plt.scatter(ell,np.median(resultso2b['X2red'],axis=1),label="o2b")
plt.ylabel(r"$\chi^2$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.show()

plt.scatter(ell,np.median(resultso1bt['w1bw1b'],axis=1),label="o1bt")
plt.ylabel(r"$w1bw1b$")
plt.xlabel(r"$\ell$")
plt.legend()
plt.show()

#plt.errorbar(ell,np.median(resultsmbb['r'],axis=1),yerr=np.std(resultsmbb['r'],axis=1),label="mbb")
#plt.errorbar(ell,np.median(resultso1b['r'],axis=1),yerr=np.std(resultso1b['r'],axis=1),label="o1b")
plt.errorbar(ell,np.median(resultso1bt['r'],axis=1),yerr=np.std(resultso1bt['r'],axis=1),label="o1bt")
plt.errorbar(ell,np.median(resultso2b['r'],axis=1),yerr=np.std(resultso2b['r'],axis=1),label="o2b")
plt.ylabel(r"$r$")
plt.xlabel(r"$\ell$")
plt.ylim([-1e-3,1e-3])
plt.legend()
plt.show()
