import sys
sys.path.append("./lib")
import numpy as np
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
import analys_lib_parallel as an

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

instr_name='litebird_full'
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

DLdc=DLdc[:50,:,:Nell]

p0=[5e2, 1.54, 20, 0] #first guess for mbb A, beta, T, r

resultsmbb_PL=np.load('Best-fits/resultsmbb_PL_d%ss%sc.npy'%(dusttype,synctype),allow_pickle=True).item()

resultso1bt_moms_full = an.fito1_bT_moms_full_parallel(nucross,DLdc,Linvdc,resultsmbb_PL,fix=0)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if synctype==None:
    np.save('Best-fits/resultso1bt_moms_full_d%sc_fix0_%s/res%s.npy'%(dusttype,rank),resultso1bt_moms_full)
else:
    np.save('Best-fits/resultso1bt_moms_full_d%ss%sc_fix0/res%s.npy'%(dusttype,synctype,rank),resultso1bt_moms_full)
