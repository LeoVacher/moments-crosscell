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
import analys_lib as an
from plotlib import plotr_gaussproduct

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

#DLdc=DLdc[:50,:,:Nell]

p0=[5e2, 1.54, 20, 10, -3,0.1, 0] #first guess for mbb A, beta, T, r

resultsmbb_PL = an.fitmbb_PL_vectorize(nucross,DLdc,Linvdc,p0,quiet=True)

# if synctype==None:
#     np.save('Best-fits/resultsmbb_PL_d%sc.npy'%dusttype,resultsmbb_PL)
# else:
#     np.save('Best-fits/resultsmbb_PL_d%ss%sc.npy'%(dusttype,synctype),resultsmbb_PL)

plotr_gaussproduct(resultsmbb_PL,Nmax=15,debug=False,color='darkorange',save=True,kwsave='MBB_PL_d%ss%s_normalize'%(synctype,dusttype))

# fit order 1 moments in beta and T around mbb pivot, get results and save

#DLdc=DLdc[:200,:,:Nell]

# p0=[100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]

#resultsmbb_PL=np.load('Best-fits/resultsmbb_PL_d%ss%sc.npy'%(dusttype,synctype),allow_pickle=True).item()

# fix=0

# resultso1bt_PL = an.fito1_bT_PL_vectorize(nucross,DLdc,Linvdc,resultsmbb_PL,quiet=True,fix=fix,fixAw=0,fixcterm=0)

# if synctype==None:
#     np.save('Best-fits/resultso1bt_PL_d%sc_fix%s_p0.npy'%(dusttype,fix),resultso1bt_PL)
# else:
#     np.save('Best-fits/resultso1bt_PL_d%ss%sc_fix%s_p0.npy'%(dusttype,synctype,fix),resultso1bt_PL)

# # # plot Gaussian likelihood for r

# plotr_gaussproduct(resultso1bt_PL,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_fullo1bT_fix%s_vectorize'%(synctype,dusttype,fix))

# resultso1bt_moms_full = an.fito1_bT_moms_full(nucross,DLdc,Linvdc,resultsmbb_PL,fix=0,quiet=False)

# if synctype==None:
#     np.save('Best-fits/resultso1bt_moms_full_d%sc_fix0.npy'%(dusttype),resultso1bt_moms_full)
# else:
#     np.save('Best-fits/resultso1bt_moms_full_d%ss%sc_fix0.npy'%(dusttype,synctype),resultso1bt_moms_full)

# plotr_gaussproduct(resultso1bt_moms_full,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_fullo1bTs_fix%s'%(dusttype,synctype,fix))
