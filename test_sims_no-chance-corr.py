import sys
sys.path.append("./lib")
from plotlib import plotrespdf
import numpy as np
import healpy as hp
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
import simu_lib as sim
import pysm3.units as u
from plotlib import plotr_gaussproduct
from plotlib import plotr_gaussproduct_analytical

r = 0
nside = 64
Npix = hp.nside2npix(nside)
N=500
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin =10
fsky = 0.7
dusttype = 1
syncrotype = 0
kw = ''
load_dust=True 
load_cmbnoise=True 

# instr param

instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
N_freqs =len(freq)
Ncross=int(N_freqs*(N_freqs+1)/2)
sens_P= instr['sens_P']
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()

#mask

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#call foreground sky

if load_dust==False:
    if dusttype==None and syncrotype==None:
        mapfg=np.zeros((N_freqs,2,Npix))
    else:
        if dusttype==None:
            sky = pysm3.Sky(nside=512, preset_strings=['s%s'%syncrotype])#,'s%s'%synctype])
        if syncrotype==None:
            sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
        if syncrotype!=None and dusttype!=None:
            sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%syncrotype])
        mapfg= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
        mapfg=mapfg[:,1:]

    #Initialise workspaces :

    wsp_dc=[]
    for i in range(0,N_freqs): 
        for j in range(i,N_freqs):
            w_dc = nmt.NmtWorkspace()
            if i != j :
                w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask,1*mapfg[j],purify_e=False, purify_b=True), b)
            if i==j :
                w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapfg[j],purify_e=False, purify_b=True), b)
            wsp_dc.append(w_dc)

    wsp_dc=np.array(wsp_dc)
    CLcross_fg_tp=np.zeros((Ncross,len(leff)))

if load_cmbnoise==False:
    CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE

    CLcross_cmb=np.zeros((N,Ncross,len(leff)))
    CLcross_noise=np.zeros((N,Ncross,len(leff)))
    for k in range(0,N):
        print('k=',k)
        noisemaps= np.zeros((3,N_freqs,2,Npix))

        
        for p in range(3):
            for i in range(N_freqs):
                noisemaps[p,i,0] =np.random.normal(0,sigpix[i],size=Npix)
                noisemaps[p,i,1] =np.random.normal(0,sigpix[i],size=Npix)

        mapcmb0= hp.synfast(CLcmb_or,nside,pixwin=False,new=True)
        mapcmb = np.array([mapcmb0 for i in range(N_freqs)])
        mapcmb = mapcmb[:,1:]

        z=0
        for i in range(0,N_freqs):
            for j in range(i,N_freqs):
                if i != j :
                    CLcross_cmb[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*mapcmb[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapcmb[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
                    CLcross_noise[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*noisemaps[0][i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*noisemaps[0][j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
                if i==j :
                    CLcross_cmb[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*mapcmb[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapcmb[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
                    CLcross_noise[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*noisemaps[1][i]*np.sqrt(2),purify_e=False, purify_b=True), nmt.NmtField(mask, 1*noisemaps[2][j]*np.sqrt(2),purify_e=False, purify_b=True), wsp_dc[z]))[3])
                z = z +1  

    CLcross_fg = leff*(leff+1)*CLcross_fg/2/np.pi
    CLcross_noise = leff*(leff+1)*CLcross_noise/2/np.pi
    CLcross_cmb = leff*(leff+1)*CLcross_cmb/2/np.pi

    np.save("./CLsimus/DLcross_fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%s"%(nside,fsky,scale,Nlbin,dusttype,syncrotype),CLcross_fg)
    np.save("./CLsimus/DLcross_noise_nside%s_fsky%s_scale%s_Nlbin%s_Gaussian"%(nside,fsky,scale,Nlbin),CLcross_noise)
    np.save("./CLsimus/DLcross_cmb_nside%s_fsky%s_scale%s_Nlbin%s"%(nside,fsky,scale,Nlbin),CLcross_cmb)


if load_dust==True:
    CLcross_fg=np.load("./CLsimus/DLcross_fg_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%s.npy"%(nside,fsky,scale,Nlbin,dusttype,syncrotype))
if load_cmbnoise==True:
    CLcross_noise=np.load("./CLsimus/DLcross_noise_nside%s_fsky%s_scale%s_Nlbin%s_Gaussian.npy"%(nside,fsky,scale,Nlbin))
    CLcross_cmb= np.load("./CLsimus/DLcross_cmb_nside%s_fsky%s_scale%s_Nlbin%s.npy"%(nside,fsky,scale,Nlbin))

Nell = len(leff)

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

#N = 500#len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.

Nfit=N
Ncov=N

DLdc=CLcross_fg + CLcross_noise+ CLcross_cmb
DLdc=DLdc[:Nfit,:,:Nell]

print(DLdc.shape)
#compute Cholesky matrix:

DL_cov = CLcross_noise[:Ncov]+ CLcross_cmb[:Ncov]

Linvdc=an.getLinvdiag(DL_cov,printdiag=True)

# fit MBB, get results and save

p0=[100, 1.54, 20, 10, -3,0, 0] #first guess for mbb A, beta, T, r

results_ds_o0 = an.fit_mom('ds_o0',nucross,DLdc,Linvdc,p0,quiet=True)

plotr_gaussproduct(results_ds_o0,Nmax=15,debug=True,color='darkorange',save=True,kwsave='d%ss%s_%s_o0_nochance-corr'%(dusttype,syncrotype,fsky))
plotr_gaussproduct_analytical(results_ds_o0,Nmax=15,debug=True,color='darkorange',save=True,kwsave='d%ss%s_%s_o0_nochance-corr'%(dusttype,syncrotype,fsky))

res1=np.load('Best-fits/results_d%ss%s_0.7_o0.npy'%(dusttype,syncrotype),allow_pickle=True).item()
res2=results_ds_o0
legs1= 'normal'
legs2= 'nocorr'

c1='darkblue'
c2='darkorange'

plotrespdf(leff,[res1,res2],[legs1,legs2],[c1,c2])
# fit order 1 moments in beta and T around mbb pivot, get results and save

#p0=[100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]

#fix=0

#results_ds_o1bt = an.fit_mom('ds_o1bt',nucross,DLdc,Linvdc,p0,quiet=True,fix=fix)

# if synctype==None:
#     np.save('Best-fits/results_d%ss_o1bt.npy'%(dusttype),results_ds_o1bt)
# else:
#     np.save('Best-fits/results_d%ss%s_%s_o1bt.npy'%(dusttype,synctype,fsky),results_ds_o1bt)

# plot Gaussian likelihood for r

#plotr_gaussproduct(results_ds_o1bt,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_%s_o1bt_nochance-corr'%(dusttype,syncrotype,fsky))

#p0=[100, 1.54, 20, 10, -3,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0]

#results_ds_o1bts = an.fit_mom('ds_o1bts',nucross,DLdc,Linvdc,p0,fix=0,quiet=False)

# if synctype==None:
#     np.save('Best-fits/results_d%s_o1bts.npy'%(dusttype),results_ds_o1bts)
# else:
#     np.save('Best-fits/results_d%ss%s_%s_o1bts.npy'%(dusttype,synctype,fsky),results_ds_o1bts)

# plot Gaussian likelihood for r

#plotr_gaussproduct(results_ds_o1bts,Nmax=15,debug=False,color='darkorange',save=True,kwsave='d%ss%s_%s_o1bts_nochance-corr'%(dusttype,synctype,fsky))
