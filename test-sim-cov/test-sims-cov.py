import sys
sys.path.append("./lib")

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

r = 0
nside = 16
Npix = hp.nside2npix(nside)
N=10000 
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
syncrotype = 0
kw = ''
load=True

# instr param

ifreq=[0,9,21]
instr_name='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
sens_P= instr['sens_P']
freq=freq[ifreq]
sens_P=sens_P[ifreq]
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()
N_freqs =len(freq)
Ncross=int(N_freqs*(N_freqs+1)/2)

#cmb
CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE

#mask

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#call foreground sky

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

np.save("./test-sim-cov/map-test/mapfg.npy",mapfg)

# call cmb

noisemaps= np.zeros((N,3,N_freqs,2,Npix))
mapcmb = np.zeros((N,N_freqs,2,Npix))

for k in range(0,N):
    print('k=',k)

    for p in range(3):
        for i in range(N_freqs):
            noisemaps[k,p,i,0] =np.random.normal(0,sigpix[i],size=Npix)
            noisemaps[k,p,i,1] =np.random.normal(0,sigpix[i],size=Npix)
    
    mapcmb0= hp.synfast(CLcmb_or,nside,pixwin=False,new=True)
    mapcmb1 = np.array([mapcmb0 for i in range(N_freqs)])
    mapcmb[k] = mapcmb1[:,1:]

np.save("./test-sim-cov/map-test/noisemaps.npy",noisemaps)
np.save("./test-sim-cov/map-test/mapcmb.npy",mapcmb)

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

def computecross(mapauto,mapcross1,mapcross2):
    CLcross=np.zeros((Ncross,len(leff)))
    z=0
    for i in range(0,N_freqs):
        for j in range(i,N_freqs):
            if i != j :
                CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*mapauto[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapauto[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            if i==j :
                CLcross[z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*mapcross1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapcross2[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            z = z +1
    return CLcross

N2=2000
CLcross_coadd= np.zeros((N2,Ncross,len(leff)))
CLcross_cmbnoise= np.zeros((N2,Ncross,len(leff)))
for k in range(0,N2):
    print(k)
    #addition du bruit aux cartes
    mapauto = mapfg  + noisemaps[k,0] + mapcmb[k]
    mapcross1 = mapfg  + noisemaps[k,1]*np.sqrt(2) + mapcmb[k]
    mapcross2 = mapfg  + noisemaps[k,2]*np.sqrt(2) + mapcmb[k]
    CLcross_coadd[k]= computecross(mapauto,mapcross1,mapcross2)
    mapauto_cmbnoise = mapfg  + noisemaps[k,0] + mapcmb[k]
    mapcross1_cmbnoise = noisemaps[k,1]*np.sqrt(2) + mapcmb[k]
    mapcross2_cmbnoise = noisemaps[k,2]*np.sqrt(2) + mapcmb[k]
    CLcross_cmbnoise[k]= computecross(mapauto_cmbnoise,mapcross1_cmbnoise,mapcross2_cmbnoise)

N3=500
cov_cmb=np.cov(np.swapaxes(CLcross_cmbnoise[:N3,:,2],0,1))
cov_coadd=np.cov(np.swapaxes(CLcross_coadd[:N3,:,2],0,1))

def plotDL(DL,l):
    c=["darkblue",'forestgreen',"darkorange","darkviolet","darkred"]
    font = {'size': 65}
    matplotlib.rc('font', **font)

    cmap   = plt.get_cmap('jet_r',402) #color map parameter
    plt.figure(figsize=(35,20))
    DL_mean=np.mean(DL, axis=0)
    DL_std=np.std(DL, axis=0)

    for i,f in enumerate(np.linspace(0,Ncross-1,Ncross)):
             plt.errorbar(l,DL_mean[int(f)],yerr=DL_std[int(f)],fmt='.',color=cmap((int(nucross[i]))),markersize=20)#,label='%s'%f)
    plt.loglog()
    plt.plot(l, DL_lens, color='black', linestyle = '--', lw=5, label='lensing',zorder=90)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=45,vmax=0))
    sm.set_array([])
    tick = 45/3
    cbar = plt.colorbar(sm, ticks=[0,tick,2*tick,3*tick])
    cbar.ax.set_yticklabels([nucross[0],np.rint(nucross[int(Ncross/3)]),np.rint(nucross[int(2*Ncross/3)]),nucross[Ncross-1]])
    cbar.set_label(r"$\sqrt{\nu_{i} \times \nu_{j}} \,\, [{\rm GHz}]$",labelpad=30)
    plt.ylabel(r"$\mathcal{D}_\ell(\nu_i \times \nu_j) \, \,  [\mu \,{\rm K}_{\rm CMB}^2]$",labelpad=56)
    plt.xlabel(r'$\ell$')
    #plt.xlim([10,205])
    plt.show()

plotDL(CLcross_cmbnoise,leff)
plotDL(CLcross_coadd,leff)

def plotcov(cov,title=''):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log10(abs(cov)), cmap='viridis', aspect='auto', vmin=-8, vmax=-5)# Set color limits
    plt.colorbar(label='$\log_{10}(|\Sigma|)$')
    plt.title(title)
    plt.xlabel(r'$\sqrt{\nu_i\nu_j}$')
    plt.ylabel(r'$\sqrt{\nu_i\nu_j}$')
    plt.tight_layout()
    plt.show()
plotcov(cov_cmb,title='cmb+n')
plotcov(cov_coadd,title='fg+cmb+n')
plt.plot(np.diag(cov_coadd))
plt.plot(np.diag(cov_cmb))
plt.show()