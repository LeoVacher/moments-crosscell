import sys
sys.path.append("./lib")

import numpy as np
import healpy as hp
import pymaster as nmt 
import pysm3
import time
from mpfit import mpfit
import scipy
#from Nearest_Positive_Definite import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import scipy.stats as st
import basicfunc as func
import analys_lib as an
import simu_lib as sim
import pysm3.units as u
from tqdm import tqdm

r = 0
nside = 32
Npix = hp.nside2npix(nside)
N=500 
lmax = nside*2-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 0
syncrotype = 0
kw = ''
load=False

# instr param

instr_name ='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
N_freqs = len(freq)
Ncross = int(N_freqs*(N_freqs+1)/2)
sens_P = instr['sens_P']
sigpix = sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()
fact_Dl= leff*(leff+1)/2/np.pi

#mask

mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))

#call foreground sky

if dusttype == None and syncrotype == None:
    mapfg=np.zeros((N_freqs,2,Npix))
else:
    if dusttype == None:
        sky = pysm3.Sky(nside=512, preset_strings=['s%s'%syncrotype])#,'s%s'%synctype])
    if syncrotype == None:
    	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
    if syncrotype != None and dusttype != None:
    	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%syncrotype])
    mapfg= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
    mapfg=mapfg[:,1:]

# call cmb

CLcmb_or = hp.read_cl('./power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE

#Initialise workspace:

wsp = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b)

#compute sims:

if load == True:
    if syncrotype == None:
        CLcross = 2*np.pi*np.load('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy'%(nside,fsky,scale,Nlbin,dusttype))/leff/(leff+1)  
    else:
        CLcross = 2*np.pi*np.load('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,syncrotype))/leff/(leff+1)  
    kini=np.argwhere(CLcross == 0)[0,0]
else:
    kini=0
    CLcross = np.zeros((N,Ncross,len(leff)))

for k in tqdm(range(kini,N)):
    noisemaps = np.zeros((3,N_freqs,2,Npix))

    for p in range(3):
        for i in range(N_freqs):
            noisemaps[p,i,0] =np.random.normal(0,sigpix[i],size=Npix)
            noisemaps[p,i,1] =np.random.normal(0,sigpix[i],size=Npix)
    
    mapcmb0 = hp.synfast(CLcmb_or,nside,pixwin=False,new=True)
    mapcmb = np.array([mapcmb0 for i in range(N_freqs)])
    mapcmb = mapcmb[:,1:]

    #add noise to maps
    maptotaldc1  = mapfg  + noisemaps[0] + mapcmb
    maptotaldc21 = mapfg  + noisemaps[1]*np.sqrt(2) + mapcmb
    maptotaldc22 = mapfg  + noisemaps[2]*np.sqrt(2) + mapcmb

    # to be replaced by sim.computecross
    CLcross[k]= sim.computecross(maptotaldc1,maptotaldc1,maptotaldc21,maptotaldc22,wsp,mask,fact_Dl=fact_Dl,coupled=False,mode='BB')

    if syncrotype==None and dusttype==None:
        if r ==0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_c"%(nside,fsky,scale,Nlbin),CLcross/2/np.pi)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_c"%(r,nside,fsky,scale,Nlbin),CLcross)
    elif syncrotype==None:
    	if r ==0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(nside,fsky,scale,Nlbin,dusttype),CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(r,nside,fsky,scale,Nlbin,dusttype),CLcross)
    elif dusttype==None:
        if r ==0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(nside,fsky,scale,Nlbin,syncrotype),CLcross)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(r,nside,fsky,scale,Nlbin,syncrotype),CLcross)
    elif syncrotype!=None and dusttype!=None:
    	if r ==0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,syncrotype),CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(r,nside,fsky,scale,Nlbin,dusttype,syncrotype),CLcross)

