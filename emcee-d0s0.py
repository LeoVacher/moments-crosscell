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
fsky = 0.8
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
kw="_bin30"

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

data= np.array([DLdc[0]])
stat= DLdc[1:]
Linvdc=an.getLinvdiag(stat,printdiag=True)

# fit MBB, get results and save

p0=[100, 1.54, 20, 10, -3,1, 0] #first guess for mbb A, beta, T, r

results_ds_o0 = an.fit_mom('ds_o0',nucross,data,Linvdc,p0,quiet=True)


DLdctempo = np.swapaxes(DLdc,0,1)
for L in range(Nell):
    covariancedc.append(np.cov(DLdctempo[:,:,2,L]))
    invcovdc.append(scipy.linalg.inv(np.array(covariancedc[L])))

def lnlike(theta, x, y, yerr,yerrminusone,l0):
    model = eml.modeldcordre0(theta,x,l0)
    chi2 = np.dot(np.dot(y-model,yerrminusone),y-model)
    return -0.5*(np.sum(chi2))

# def lnprior(theta):
    # Amp, beta, r = theta
    # if -np.inf < Amp < np.inf and -np.inf < beta < np.inf and -np.inf < r < np.inf:
    #     return 0.0
    # return -np.inf

def lnprob(theta, x, y, yerr,yerrminusone,l0):
    # lp = lnprior(theta)
    # if not np.isfinite(lp):
    #     return -np.inf
    return  lnlike(theta, x, y, yerr,yerrminusone,l0) #+lp

n = 0
L = 0

mcmcparamiterl = []#list(np.load("/home/lvacher/emceefit/mcmcparamiterld0c.npy"))
chi2l = []# list(np.load("/home/lvacher/emceefit/mcmcchi2ld0c.npy"))

ini = len(mcmcparamiterl)

for L in range(ini,Nell):
    mcmcparamitern = []
    chi2n=[]
    for n in range(N):
        time1 = time.time()
        y = DLdc[n,:,2,L]
        #yerr = DLstddc[:,2,L]
        yerr =  covariancedc[L]
        yerrminusone = invcovdc[L]
        pl0=[AmpTrue,betaTrue,rTrue,L,temp_dust]
        parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0},{'value':pl0[2], 'fixed':0},{'value':pl0[3], 'fixed':1},{'value':pl0[4], 'fixed':1}]
        fa = {'x':nucross, 'y':DLdc[n,:,2,L], 'err': Linvdc[L]}
        m = mpfit(mpl.Fitdcordre0,parinfo= parinfopl ,functkw=fa)
        thetafit = [m.params[0], m.params[1],m.params[2]]
        ndim, nwalkers,chainlength,burnt = 3, 6, 600,200
        pos = [np.array(thetafit) + 1e-6*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr,yerrminusone,L))
        sampler.run_mcmc(pos, chainlength)
        samples = sampler.get_chain()
        samples = sampler.chain[:, burnt:, :].reshape((-1, ndim))

        amp_mcmc, beta_mcmc, r_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
        theta_mcmc = np.array([amp_mcmc[0],beta_mcmc[0],r_mcmc[0]])
        mcmcparamitern.append(np.array([amp_mcmc,beta_mcmc,r_mcmc]))
        print('n=',n)
        print('L=',L)
        model = eml.modeldcordre0(theta_mcmc,x,L)
        chi2tempo = np.dot(np.dot(y-model,yerrminusone),y-model)
        chi2n.append(chi2tempo)

        time2 = time.time()
        print(time2-time1)
    mcmcparamiterl.append(np.array(mcmcparamitern))
    chi2l.append(np.array(chi2n))
    np.save("/home/lvacher/emceefit/mcmcparamiterld0c1o0",mcmcparamiterl)
    np.save("/home/lvacher/emceefit/mcmcchi2ld0c1o0",chi2l)
