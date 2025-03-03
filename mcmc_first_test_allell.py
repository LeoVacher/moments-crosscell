import sys
sys.path.append("./lib")
import numpy as np
import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import covlib as cvl 
import emcee
import fitlib as ftl 
from getdist import plots, MCSamples
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
Pathload='./'
all_ell=True #all ell or each ell independently
fix= 0 #fix beta and T ?
adaptative=False
N=500
parallel=False
cov_type='sim' #choices: sim, Knox-fg, Knox+fg, signal.
kw=''
if cov_type!='sim':
    kw+='_%s'%cov_type

if parallel==True:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

# Call C_ell of simulation

if synctype==None:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy"%(nside,fsky,scale,Nlbin,dusttype))
else:
    DLdc = np.load(Pathload+"/CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

DLdc=DLdc[:N,:,:ELLBOUND]

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

ncross=len(nucross)
nnus = int((-1 + np.sqrt(ncross * 8 + 1)) / 2.)
posauto = [int(nnus * i - i * (i + 1) / 2 + i) for i in range(nnus)]
nu = nucross[posauto]
freq_pairs = np.array([(i, j) for i in range(nnus) for j in range(i, nnus)])
nu_i = nu[freq_pairs[:, 0]]
nu_j = nu[freq_pairs[:, 1]]

# fit MBB and PL, get results, save and plot

n=0

DL_lensbin, DL_tens= ftl.getDL_cmb(nside=nside,Nlbin=Nlbin)

res1=np.load('Best-fits/results_d%ss%s_0.7_ds_o0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()
p0_ell = []
p0_ell.extend(res1['A'][:,n]) 
p0_ell.extend(res1['A_s'][:,n]) 
p0_ell.extend(res1['A_sd'][:,n])
p0_ell.append(1.5) #betad
p0_ell.append(1/20) #1/Td
p0_ell.append(-3) #betas    
p0_ell.append(0.0001) #r 
p0_ell = np.array(p0_ell)

#put arrays in NcrossxNell shape for all-ell fit
nu_i = np.tile(nu_i, Nell)
nu_j = np.tile(nu_j, Nell)
DL_lensbin= np.repeat(DL_lensbin[:Nell],ncross)
DL_tens= np.repeat(DL_tens[:Nell],ncross)
DLdcflat = np.zeros([N,Nell*ncross])
for i in range(N):
    DLdcflat[i] = np.swapaxes(DLdc[i,:,:],0,1).flatten()

y = DLdcflat[n]
yerr1 = np.cov(np.swapaxes(DLdcflat[:,:],0,1))
invcov1 = np.linalg.inv(yerr1)

thetafit = p0_ell
ndim, nwalkers,chainlength,burnt= len(thetafit), 2*len(thetafit), 30000*len(thetafit), int(0.1*30000*len(thetafit))
pos = [np.array(thetafit) + 1e-4*np.random.randn(ndim)*np.array(thetafit) for i in range(nwalkers)]

model = ftl.func_ds_o0_all_ell(np.array(p0_ell), x1=nu_i, x2=nu_j,nuref=353.,nurefs=23.,DL_lensbin=DL_lensbin, DL_tens=DL_tens,Nell=Nell)

sampler1 = emcee.EnsembleSampler(
    nwalkers, ndim, ftl.lnprob, 
    args=(y, invcov1),  
    kwargs={                  
        "model_func": ftl.func_ds_o0_all_ell,  
        "x1": nu_i, 
        "x2": nu_j, 
        "nuref": 353., 
        "nurefs": 23., 
        "DL_lensbin": DL_lensbin, 
        "DL_tens": DL_tens,
        "Nell": Nell,
        "all_ell": True
    })

def runsample(sampler,invcov):
    sampler.run_mcmc(pos, chainlength, progress=True)
    samples = sampler.get_chain()
    samples = sampler.chain[:, burnt:, :].reshape((-1, ndim))
    flat_samples = sampler.get_chain(discard=int(0.1*chainlength), thin=15, flat=True)
    return flat_samples

flat_samples1=runsample(sampler1,invcov1)
np.save("./samples/sample_test_d0s0.npy",flat_samples1)

param_names = []
[param_names.append("A^d_%s"%i) for i in range(Nell)]
[param_names.append("A^s_%s"%i) for i in range(Nell)]
[param_names.append("A^{sd}_%s"%i) for i in range(Nell)]
param_names.append("\\beta_d") 
param_names.append("1/T_d") 
param_names.append("\\beta_s") 
param_names.append("r") 

samples1 = MCSamples(samples=flat_samples1[:,(ndim-4):], names=param_names[(ndim-4):], labels=param_names[(ndim-4):])
g = plots.get_subplot_plotter()
g.triangle_plot(samples1, 
                filled=True, 
                contour_colors="darkblue", 
                contour_levels=[0.68, 0.95],
                title_limit=1,
                )
plt.savefig("./corner_plots/mcmc_v1.pdf")
plt.show()

#corner.corner(flat_samples1,labels=param_names,smooth=0.5,color="darkred",truths= p0_ell,truth_color='r',show_titles=True,quantiles=[0.16,0.5,0.84], math_text = True,title='Knox+fg',opacity=0.1,plot_datapoints=False,fill_contours=True,levels=(0.68, 0.95))
