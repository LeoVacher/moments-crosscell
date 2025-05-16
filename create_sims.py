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
nside = 64
Npix = hp.nside2npix(nside)
N=500 
lmax = nside*3-1
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 10
syncrotype = 4
kw = ''
load=False
masking_strat='GWD'

if masking_strat=='GWD':
    kw = kw + '_maskGWD'

# instr param

instr_name ='litebird_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
N_freqs = len(freq)
Ncross = int(N_freqs*(N_freqs+1)/2)
sens_P = instr['sens_P']
sigpix = sens_P/hp.nside2resol(nside, arcmin=True)
b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
leff = b.get_effective_ells()
Nell = len(leff)

#call foreground sky

if dusttype == None and syncrotype == None:
    mapfg = np.zeros((N_freqs,2,Npix))
else:
    if dusttype == None:
        sky = pysm3.Sky(nside=512, preset_strings=['s%s'%syncrotype])#,'s%s'%synctype])
    if syncrotype == None:
    	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
    if syncrotype != None and dusttype != None:
    	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%syncrotype])
    mapfg = np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
    mapfg = mapfg[:,1:]

# call cmb

CLcmb_or = hp.read_cl('./power_spectra/Cls_Planck2018_r0.fits') #TT EE BB TE


#mask

if masking_strat=='maskGWD':
    mask = masks_WGD(abs(mapfg[-1,1]+1j*mapfg[-1,2]), 
          per_cent_to_keep = fsky*100, 
          smooth_mask_deg = 2, 
          apo_mask_deg = scale, 
          verbose=False)
else:
    if fsky==1:
        mask=np.ones(Npix)
    else:
        mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))


#Initialise workspace:

wsp = sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b)

#compute sims:

if load == True:
    if syncrotype == None:
        CLcross = np.load('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc.npy'%(nside,fsky,scale,Nlbin,dusttype))
    else:
        CLcross = np.load('./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc.npy'%(nside,fsky,scale,Nlbin,dusttype,syncrotype))  
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

    CLcross[k]= sim.computecross(maptotaldc1,maptotaldc1,maptotaldc21,maptotaldc22,wsp,mask,Nell,b,coupled=False,mode='BB')

    if syncrotype==None and dusttype==None:
        if r == 0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_c"%(nside,fsky,scale,Nlbin),CLcross)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_c"%(r,nside,fsky,scale,Nlbin),CLcross)
    elif syncrotype==None:
    	if r == 0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(nside,fsky,scale,Nlbin,dusttype),CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(r,nside,fsky,scale,Nlbin,dusttype),CLcross)
    elif dusttype==None:
        if r == 0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(nside,fsky,scale,Nlbin,syncrotype),CLcross)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(r,nside,fsky,scale,Nlbin,syncrotype),CLcross)
    elif syncrotype!=None and dusttype!=None:
    	if r == 0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,syncrotype),CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(r,nside,fsky,scale,Nlbin,dusttype,syncrotype),CLcross)

