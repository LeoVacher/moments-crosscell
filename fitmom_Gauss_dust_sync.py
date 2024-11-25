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
from plotlib import plotrespdf

r=0.
nside = 64
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
ELLBOUND = 15
dusttype = 1
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
kw=""

#compute cross-frequencies 

nucross = []
for i in range(0,nf):
    for j in range(i,nf):
        nucross.append(np.sqrt(freq[i]*freq[j]))
nucross = np.array(nucross)

#N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.
N=500

DLdc=DLdc[:N,:,:Nell]

#compute Cholesky matrix:

Linvdc=an.getLinvdiag(DLdc,printdiag=True)

# fit MBB, get results, save and plot

p0=[100, 1.54, 20, 10, -3,0, 0] #first guess for mbb A, beta, T, r
results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin)
if synctype==None:
    np.save('Best-fits/results_d%s_o0%s.npy'%(dusttype,kw),results_ds_o0)
    plotrespdf(l,[results_ds_o0],['d%s-o0'%(dusttype)],['darkorange'])
    plotr_gaussproduct(results_ds_o0,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%s_%s_o0%s'%(dusttype,fsky,kw))
else:
    np.save('Best-fits/results_d%ss%s_%s_o0%s.npy'%(dusttype,synctype,fsky,kw),results_ds_o0)
    plotrespdf(l,[results_ds_o0],['d%ss%s-o0'%(dusttype,synctype)],['darkorange'])
    plotr_gaussproduct(results_ds_o0,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_%s_o0%s'%(dusttype,synctype,fsky,kw))

# fit order 1 in beta and T, get results, save and plot

p0=[100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin)
if synctype==None:
    np.save('Best-fits/results_d%ss_o1bt%s.npy'%(dusttype,kw),results_ds_o1bt)
    plotrespdf(l,[results_ds_o1bt],['d%s-o1bt'%(dusttype)],['darkorange'])
    plotr_gaussproduct(results_ds_o1bt,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%s_%s_o1bt%s'%(dusttype,fsky,kw))
else:
    np.save('Best-fits/results_d%ss%s_%s_o1bt%s.npy'%(dusttype,synctype,fsky,kw),results_ds_o1bt)
    plotrespdf(l,[results_ds_o1bt],['d%ss%s-o1bt'%(dusttype,synctype)],['darkorange'])
    plotr_gaussproduct(results_ds_o1bt,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_%s_o1bt%s'%(dusttype,synctype,fsky,kw))

# fit order 1 in beta, T and beta_s, get results, save and plot

p0=[100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]
results_ds_o1bts = an.fit_mom('ds_o1bts',nucross,DLdc,Linvdc,p0,quiet=True,nside=nside, Nlbin=Nlbin)
if synctype==None:
    np.save('Best-fits/results_d%s_o1bts%s.npy'%(dusttype,kw),results_ds_o1bts)
    plotrespdf(l,[results_ds_o1bts],['d%ss-o1bts'%(dusttype)],['darkorange'])
    plotr_gaussproduct(results_ds_o1bts,Nmax=6,debug=False,color='darkorange',save=True,kwsave='d%s_%s_o1bts%s'%(dusttype,fsky,kw))
else:
    np.save('Best-fits/results_d%ss%s_%s_o1bts%s.npy'%(dusttype,synctype,fsky,kw),results_ds_o1bts)
    plotrespdf(l,[results_ds_o1bts],['d%ss%s-o1bts'%(dusttype,synctype)],['darkorange'])
    plotr_gaussproduct(results_ds_o1bts,Nmax=6,debug=False,color='darkorange',save=True,kwsave='d%ss%s_%s_o1bts%s'%(dusttype,synctype,fsky,kw))


