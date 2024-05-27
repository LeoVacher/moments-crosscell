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

# Call Cell of simulation

if synctype==None:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy"%(nside,fsky,scale,Nlbin,dusttype))
else:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
l = b.get_effective_ells()
l = l[:ELLBOUND]
Nell = len(l)

#instrument informations:

instr_name='LiteBIRD_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
freq=freq
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

Linvdc=an.getLinvdiag(DLdc,printdiag=True)

# fit MBB, get results and save

#DLdc=DLdc[:200,:,:Nell]

p0=[5e2, 1.54, 20, 0] #first guess for mbb A, beta, T, r

resultsmbb = an.fitmbb(nucross,DLdc,Linvdc,p0)

if synctype==None:
    np.save('Best-fits/resultsmbb_d%sc.npy'%dusttype,resultsmbb)
else:
    np.save('Best-fits/resultsmbb_d%ss%sc.npy'%(dusttype,synctype),resultsmbb)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultsmbb,Nmax=15,debug=True,color='darkorange')

resultsmbb=np.load('Best-fits/resultsmbb_d%sc.npy'%dusttype,allow_pickle=True).item()

# fit order 1 moments in beta and T around mbb pivot, get results and save

resultso1bt = an.fito1_bT(nucross,DLdc,Linvdc,resultsmbb)

if synctype==None:
    np.save('Best-fits/resultso1bt_d%sc.npy'%dusttype,resultso1bt)
else:
    np.save('Best-fits/resultso1bt_d%ss%sc.npy'%(dusttype,synctype),resultso1bt)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultso1bt,Nmax=15,debug=True,color='darkorange')
