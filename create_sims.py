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

r = 0 # input tensor-to-scalar ratio in the simulation
nside = 64 #HEALPix nside
Npix = hp.nside2npix(nside) #number of pixels
N=5000  #number of sims
lmax = nside*3-1 #maximum multipole
scale = 10 #apodization scale in degrees
Nlbin = 10 #binning scheme of the Cls
fsky = 0.7 #fraction of sky for the raw mask
dusttype = 1 #Pysm dust model
syncrotype = 1 #Pysm syncrotron model
kws = '_N5000' #keyword for the simulation
load=False #load previous sims 
masking_strat='' #keywords for choice of mask. If '', use Planck mask 
gaussbeam = True #smooth with gaussian beam?

if masking_strat=='GWD': #masking strategy From Gilles Weyman Depres (test)
    kws = kws + '_maskGWD'

# instr param

instr_name ='litebird_full' #instrument name in ./lib/instr_dict/
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq = instr['frequencies']
N_freqs = len(freq)
Ncross = int(N_freqs*(N_freqs+1)/2)
sens_P = instr['sens_P']
beam = instr['beams']
sigpix = sens_P/hp.nside2resol(nside, arcmin=True)
b = nmt.NmtBin.from_lmax_linear(lmax=lmax,nlb=Nlbin,is_Dell=True)
leff = b.get_effective_ells()
Nell = len(leff)

if gaussbeam:
    kws = kws + '_gaussbeam'
    Bls = np.zeros((N_freqs, 3*nside))
    for i in range(N_freqs):
         Bls[i] = hp.gauss_beam(beam[i], lmax=3*nside-1, pol=True).T[2]

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
          gaussbeam_mask_deg = 2, 
          apo_mask_deg = scale, 
          verbose=False)
else:
    if fsky==1:
        mask=np.ones(Npix)
    else:
        mask = hp.read_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale))


#Initialise workspace:

if gaussbeam:
     wsp = []
     for i in range(N_freqs):
         for j in range(i, N_freqs):
            wsp.append(sim.get_wsp(mapfg,mapfg,mapfg,mapfg,mask,b,purify='BB', beam1=Bls[i], beam2=Bls[j]))
else:
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

    #create three random noises corresponding to full mission and the two half-missions
    for p in range(3):
        for i in range(N_freqs):
            noisemaps[p,i,0] =np.random.normal(0,sigpix[i],size=Npix) #Q
            noisemaps[p,i,1] =np.random.normal(0,sigpix[i],size=Npix) #U
    
    mapcmb0 = hp.synfast(CLcmb_or,nside,pixwin=False,new=True)
    mapcmb = np.array([mapcmb0 for i in range(N_freqs)])
    mapcmb = mapcmb[:,1:]

    # Sky signal
    signal = mapfg + mapcmb

    if gaussbeam:
        for i in range(N_freqs):
            for j in range(2): #smooth Q and U maps
                signal[i,j] = hp.smoothing(signal[i,j], fwhm=beam[i])

    #add noise to maps
    maptotaldc1  = signal + noisemaps[0]
    maptotaldc21 = signal + noisemaps[1]*np.sqrt(2)
    maptotaldc22 = signal + noisemaps[2]*np.sqrt(2)

    CLcross[k]= sim.computecross(maptotaldc1,maptotaldc1,maptotaldc21,maptotaldc22,wsp,mask,Nell,b,coupled=False,mode='BB')

    #save:
    if syncrotype==None and dusttype==None:
        if r == 0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_c"%(nside,fsky,scale,Nlbin)+kws,CLcross)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_c"%(r,nside,fsky,scale,Nlbin)+kws,CLcross)
    elif syncrotype==None:
    	if r == 0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(nside,fsky,scale,Nlbin,dusttype)+kws,CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%sc"%(r,nside,fsky,scale,Nlbin,dusttype)+kws,CLcross)
    elif dusttype==None:
        if r == 0:
            np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(nside,fsky,scale,Nlbin,syncrotype)+kws,CLcross)
        else :
            np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_s%sc"%(r,nside,fsky,scale,Nlbin,syncrotype)+kws,CLcross)
    elif syncrotype!=None and dusttype!=None:
    	if r == 0:
    		np.save("./power_spectra/DLcross_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(nside,fsky,scale,Nlbin,dusttype,syncrotype)+kws,CLcross)
    	else :
    		np.save("./power_spectra/DLcross_r%s_nside%s_fsky%s_scale%s_Nlbin%s_d%ss%sc"%(r,nside,fsky,scale,Nlbin,dusttype,syncrotype)+kws,CLcross)

