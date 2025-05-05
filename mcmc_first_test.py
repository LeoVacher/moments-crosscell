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
lmax = nside*2-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
synctype = 0
Pathload='./'
all_ell=False #all ell or each ell independently
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

# Initialize binning scheme with Nlbin ells per bandpower

b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
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
L=3

DL_lensbin, DL_tens= ftl.getDL_cmb(nside=nside,Nlbin=Nlbin)

res1=np.load('best_fits/results_d%ss%s_0.7_ds_o0_fix%s.npy'%(dusttype,synctype,fix),allow_pickle=True).item()
p0=[]
for k in range(len(res1.keys())-1):
    if list(res1.keys())[k]=='temp':
        p0.append(1/res1[list(res1.keys())[k]][L,n])
    else:
        p0.append(res1[list(res1.keys())[k]][L,n])

y = DLdc[n,:,L]
cov1 = np.cov(np.swapaxes(DLdc[:,:,L],0,1))
cov2 = np.load(Pathload+"/covariances/cov_Knox-fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))
cov3 = np.load(Pathload+"/covariances/cov_Knox+fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy"%(nside,fsky,scale,Nlbin,dusttype,synctype))

invcov1 = np.linalg.inv(yerr1)
invcov2= cvl.inverse_covmat(cov2, Ncross, neglect_corbins=True, return_cholesky=False, return_new=False)[L]
invcov3= cvl.inverse_covmat(cov3, Ncross, neglect_corbins=True, return_cholesky=False, return_new=False)[L]

thetafit = p0
model0 = ftl.func_ds_o0(p0, x1=nu_i, x2=nu_j,nuref=353.,nurefs=23.,ell=L,DL_lensbin=DL_lensbin, DL_tens=DL_tens)
ndim, nwalkers,chainlength,burnt= len(p0), 2*len(p0), 2000*len(p0), int(0.1*2000*len(p0))
pos = [np.array(thetafit) + 1e-2*np.random.randn(ndim)*np.array(thetafit) for i in range(nwalkers)]

sampler1 = emcee.EnsembleSampler(
    nwalkers, ndim, ftl.lnprob, 
    args=(y, invcov1),  
    kwargs={                
        "model_func": ftl.func_ds_o0,  
        "x1": nu_i, 
        "x2": nu_j, 
        "nuref": 353., 
        "nurefs": 23., 
        "ell": L, 
        "DL_lensbin": DL_lensbin, 
        "DL_tens": DL_tens,
        "all_ell": False
    })
sampler2 = emcee.EnsembleSampler(
    nwalkers, ndim, ftl.lnprob, 
    args=(y, invcov2),  
    kwargs={                  
        "model_func": ftl.func_ds_o0,  
        "x1": nu_i, 
        "x2": nu_j, 
        "nuref": 353., 
        "nurefs": 23., 
        "ell": L, 
        "DL_lensbin": DL_lensbin, 
        "DL_tens": DL_tens,
        "all_ell": False
    })
sampler3 = emcee.EnsembleSampler(
    nwalkers, ndim, ftl.lnprob, 
    args=(y, invcov3),  
    kwargs={                  
        "model_func": ftl.func_ds_o0,  
        "x1": nu_i, 
        "x2": nu_j, 
        "nuref": 353., 
        "nurefs": 23., 
        "ell": L, 
        "DL_lensbin": DL_lensbin, 
        "DL_tens": DL_tens,
        "all_ell": False
    })

def chi2(sampler,invcov):
    """
    get chi2 from sample
    """
    samples = sampler.get_chain()
    samples = sampler.chain[:, burnt:, :].reshape((-1, ndim))
    amp_mcmc, beta_mcmc, temp_mcmc, A_s_mcmc, beta_s_mcmc, A_sd_mcmc,  r_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
    theta_mcmc = np.array([amp_mcmc[0], beta_mcmc[0], temp_mcmc[0], A_s_mcmc[0], beta_s_mcmc[0], A_sd_mcmc[0],  r_mcmc[0] ])
    model0 = ftl.func_ds_o0(p0, x1=nu_i, x2=nu_j,nuref=353.,nurefs=23.,ell=L,DL_lensbin=DL_lensbin, DL_tens=DL_tens)
    model = ftl.func_ds_o0(theta_mcmc, x1=nu_i, x2=nu_j,nuref=353.,nurefs=23.,ell=L,DL_lensbin=DL_lensbin, DL_tens=DL_tens)
    dof=Ncross-len(p0)
    chi2 = np.dot(np.dot(y-model,invcov),y-model)/dof
    print('chi2=%s'%chi2)

def run_mcmc_with_convergence(sampler, pos, max_iter=chainlength, min_iter=100, check_interval=100, tol=1e-4):
    """
    Run mcmc and stop when convergence is reached.
    """
    index = 0
    old_tau = np.inf  
    for sample in sampler.sample(pos, iterations=max_iter, progress=True):
        index += check_interval
        if index < min_iter:
            continue  
        
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            continue  

        if np.all(np.abs(tau - old_tau) / old_tau < tol):
            print(f"Convergence reached after {index} iterations.")
            break
        old_tau = tau  

    return sampler.get_chain(flat=True,discard=burnt)

flat_samples1 = run_mcmc_with_convergence(sampler1, pos)
flat_samples2 = run_mcmc_with_convergence(sampler2, pos)
flat_samples3 = run_mcmc_with_convergence(sampler3, pos)

chi2(sampler1,invcov1)
chi2(sampler2,invcov2)
chi2(sampler3,invcov3)

param_names = ["A_d", "\\beta_d", "1/T_d", "A_s", "\\beta_s", "A_{sd}", "r"]
samples1 = MCSamples(samples=flat_samples1, names=param_names, labels=param_names)
samples2 = MCSamples(samples=flat_samples2, names=param_names, labels=param_names)
samples3 = MCSamples(samples=flat_samples3, names=param_names, labels=param_names)
g = plots.get_subplot_plotter()
g.settings.alpha_filled_add=0.6
g.triangle_plot([samples1, samples2, samples3], 
                filled=True, 
                contour_colors=["darkblue", "darkorange", "darkred"], 
                contour_levels=[0.68, 0.95],
                title_limit=1,
                legend_labels=["Sim", "Knox-fg", "Knox+fg"],
                )
plt.savefig("./corner_plots/mcmc_v0.pdf")
plt.show()
