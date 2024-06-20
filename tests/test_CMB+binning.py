import sys
sys.path.append("./lib")

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
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
import analys_lib as an
import simu_lib as sim
import pysm3.units as u

r = 0
nside = 64
Npix = hp.nside2npix(nside)
N=500 
lmax = nside*3-1
#lmax=850
scale = 10
Nlbin = 10
fsky = 0.7
dusttype = 1
syncrotype = 1
mascut=0
kw = ''

# instr param

instr_name='LiteBIRD_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
N_freqs =len(freq)
Ncross=int(N_freqs*(N_freqs+1)/2)
sens_P= instr['sens_P']
sigpix= sens_P/(np.sqrt((4*np.pi)/Npix*(60*180/np.pi)**2))
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()

#mask

cut = {0.5:53,0.6:80,0.7:121}
mask0 = hp.read_map("./masks/mask_nside512_fsky%spc_P353_smooth10deg_cut%smuK.fits"%(np.int(fsky*100),cut[fsky]))
mask0 = hp.ud_grade(mask0,nside_out=nside)
mask = nmt.mask_apodization(mask0, scale, apotype='C2')

CLcmb_or=hp.read_cl('./CLsimus/Cls_Planck2018_r0.fits') #TT EE BB TE

mapcmb0= hp.synfast(CLcmb_or,nside,pixwin=False,new=True)
wsp_dc = nmt.NmtWorkspace()
wsp_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapcmb0[1:],purify_e=False, purify_b=True), nmt.NmtField(mask,1*mapcmb0[1:],purify_e=False, purify_b=True), b)

CLsimus=np.zeros((N,len(leff)))
for k in range(0,N):
    print('k=',k)
    mapcmb0= hp.synfast(CLcmb_or,nside,pixwin=False,new=True)

    CLsimus[k]=np.array((sim.compute_master(nmt.NmtField(mask, 1*mapcmb0[1:],purify_e=False, purify_b=True), nmt.NmtField(mask, mapcmb0[1:],purify_e=False, purify_b=True), wsp_dc))[3])

CL_tens=hp.read_cl('./CLsimus/Cls_Planck2018_tensor_r1.fits')

ELLBOUND=20
DL_lensbin = leff*(leff+1)*b.bin_cell(CLcmb_or[2,2:lmax+3])[0:ELLBOUND]/2/np.pi
DL_lensbin2 = leff*(leff+1)*b.bin_cell(CLcmb_or[2,0:lmax+1])[0:ELLBOUND]/2/np.pi
DL_tens = leff*(leff+1)*b.bin_cell(CL_tens[2,2:lmax+3])[0:ELLBOUND]/2/np.pi
DL_simus= leff*(leff+1)*np.mean(CLsimus,axis=0)/2/np.pi
DL_tens2 = leff*(leff+1)*b.bin_cell(CL_tens[2,0:lmax+1])[0:ELLBOUND]/2/np.pi
DL_lensbin3 = b.bin_cell(np.load("./CLsimus/DLth_CAMBparamPlanck2018_ajust.npy")[2,0:lmax+1])[0:ELLBOUND]
DL_tens3 = b.bin_cell(np.load("./CLsimus/DLtensor_CAMBparamPlanck2018_r=1.npy")[2,0:lmax+1])[0:ELLBOUND]
 

plt.plot(ell_th,ell_th*(ell_th+1)*CL_tens[2]/2/np.pi,label='th')
plt.plot(leff,DL_tens,label='2+')
plt.plot(leff,DL_tens2,label='0+')
plt.plot(leff,DL_tens3,label='old')
plt.legend()
plt.show()

plt.plot(ell_th,ell_th*(ell_th+1)*CLcmb_or[2]/2/np.pi,label='th')
plt.plot(leff,DL_simus,label='sim')
plt.plot(leff,DL_lensbin,label='lens 2+')
plt.plot(leff,DL_lensbin2,label='sim 0+')
plt.plot(leff,DL_lensbin3,label='old')
plt.legend()
plt.show()
