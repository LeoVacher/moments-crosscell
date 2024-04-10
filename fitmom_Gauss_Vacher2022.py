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
nside = 256
lmax = nside*3-1
#lmax=850
scale = 5
Nlbin = 10
fsky = 0.7
ELLBOUND = 20
dusttype = 1
kw=''
kwsim=''
Pathload='./'

# Call Cell of simulation

DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc_nobeam%s.npy"%(nside,fsky,scale,Nlbin,dusttype,kwsim))

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

#instrument informations:

instr_name='LiteBIRD_reduced'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
freq = freq[6:]
nf = len(freq)
Ncross = int(nf*(nf+1)/2)

#compute cross-frequencies 

nucross = []
for i in range(0,nf):
    for j in range(i,nf):
        nucross.append(np.sqrt(freq[i]*freq[j]))
nucross = np.array(nucross)

N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.

DLdc=DLdc[:N,:,:Nell]

#compute Cholesky matrix:

Linvdc=an.getLinvdiag(DLdc)

# fit MBB, get results and save

p0=[5e2, 1.54, 20, 0] #first guess for mbb A, beta, T, r

resultsmbb = an.fitmbb(nucross,DLdc,Linvdc,p0)
np.save('Best-fits/resultsmbb_d%sc.npy'%dusttype,resultsmbb)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultsmbb)

# fit order 1 moments in beta around mbb pivot, get results and save

resultso1b = an.fito1_b(nucross,DLdc,Linvdc,resultsmbb)
np.save('Best-fits/resultso1b_d%sc.npy'%dusttype,resultso1b,allow_pickle=True)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultso1b)

# fit order 1 moments in beta and T around mbb pivot, get results and save

resultso1bt = an.fito1_bT(nucross,DLdc,Linvdc,resultsmbb)
np.save('Best-fits/resultso1bt_d%sc.npy'%dusttype,resultso1bt)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultso1bt)

resultso2b = an.fito2_b(nucross,DLdc,Linvdc,resultsmbb)
np.save('Best-fits/resultso2b_d%sc.npy'%dusttype,resultso2b)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultso2b)

