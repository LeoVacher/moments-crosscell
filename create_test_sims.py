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
nside = 64
Npix = hp.nside2npix(nside)
N=10000 
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = None
syncrotype = None
kw = ''

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

# call cmb

CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE

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

CLcross=np.zeros((N,Ncross,len(leff)))
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

    #addition du bruit aux cartes
    maptotaldc1 = mapfg  + noisemaps[0] + mapcmb
    maptotaldc21 = mapfg  + noisemaps[1]*np.sqrt(2) + mapcmb
    maptotaldc22 = mapfg  + noisemaps[2]*np.sqrt(2) + mapcmb

    #gérer list et concatenate

    z=0
    for i in range(0,N_freqs):
        for j in range(i,N_freqs):
            if i != j :
                CLcross[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*maptotaldc1[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*maptotaldc1[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            if i==j :
                CLcross[k,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*maptotaldc21[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*maptotaldc22[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
            z = z +1  
    if syncrotype==None and dusttype==None:
        if r ==0:
            np.save("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_c"%(nside,fsky,scale,Nlbin),leff*(leff+1)*CLcross/2/np.pi)
        else :
            np.save("./CLsimus/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_c"%(r,nside,fsky,scale,Nlbin),leff*(leff+1)*CLcross/2/np.pi)
    if syncrotype==None:
    	if r ==0:
    		np.save("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(nside,fsky,scale,Nlbin,dusttype),leff*(leff+1)*CLcross/2/np.pi)
    	else :
    		np.save("./CLsimus/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(r,nside,fsky,scale,Nlbin,dusttype),leff*(leff+1)*CLcross/2/np.pi)
    if dusttype==None:
        if r ==0:
            np.save("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(nside,fsky,scale,Nlbin,syncrotype),leff*(leff+1)*CLcross/2/np.pi)
        else :
            np.save("./CLsimus/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(r,nside,fsky,scale,Nlbin,syncrotype),leff*(leff+1)*CLcross/2/np.pi)
    if syncrotype!=None and dusttype!=None:
    	if r ==0:
    		np.save("./CLsimus/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,syncrotype),leff*(leff+1)*CLcross/2/np.pi)
    	else :
    		np.save("./CLsimus/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(r,nside,fsky,scale,Nlbin,dusttype,syncrotype),leff*(leff+1)*CLcross/2/np.pi)

