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
from plotlib import getr_gaussproduct
import matplotlib.pyplot as plt 

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

# Call C_ell of simulation

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

Nlist = [300,500,700,1000,2500,5000]#,10000]

r_N=[]
sig_N=[]

DLdc=DLdc[:500,:,:Nell]

for N in Nlist:
    #compute Cholesky matrix:

    Linvdc=an.getLinvdiag(DLdc[:N,:,:Nell],printdiag=True)

    # fit MBB, get results and save

    p0=[100, 1.54, 20, 10, -3,1, 0] #first guess for mbb A, beta, T, r

    results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True)

    mu, sigma = getr_gaussproduct(results_ds_o0,Nmax=15)
    r_N.append(mu)
    sig_N.append(sigma)

Nlist=np.array(Nlist)
r_N=np.array(r_N)
sig_N=np.array(sig_N)

plt.errorbar(Nlist,r_N,yerr=sig_N)
plt.xlabel(r"$N$")
plt.ylabel(r"$r\pm\sigma(r)$")
plt.savefig("./pdf_plots/cov_nsim_test.pdf")