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

Linvdc2=np.zeros(Linvdc.shape)
for ell in range(Nell):
    np.fill_diagonal(Linvdc2[ell], 1/np.std(DLdc[:,:,ell],axis=0))
# fit MBB, get results and save

Linvdc3=np.zeros(Linvdc.shape)

for ell in range(Nell):
    Linvdc3[ell]=np.diag(np.diag(Linvdc[ell]/10))

Linvdc4 = Linvdc
for ell in range(Nell):
    np.fill_diagonal(Linvdc4[ell],2*np.diag(Linvdc[ell]))

DLdc=DLdc[:50,:,:Nell]

p0=[5e2, 1.54, 20, 0] #first guess for mbb A, beta, T, r

resultsmbb = an.fitmbb(nucross,DLdc,Linvdc4,p0)

# plot Gaussian likelihood for r

an.plotr_gaussproduct(resultsmbb,Nmax=15,debug=True,color='darkorange')
