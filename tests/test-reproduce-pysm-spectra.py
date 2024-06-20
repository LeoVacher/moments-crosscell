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
scale = 5
Nlbin = 10
fsky = 0.7
dusttype = 0
syncrotype = None
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

#call foreground sky

if syncrotype==None:
	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
else:
	sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype,'s%s'%syncrotype])

mapfg= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
mapfg=mapfg[:,1:]

wsp_dc=[]
for i in range(0,N_freqs): 
    for j in range(i,N_freqs):
        w_dc = nmt.NmtWorkspace()
        if i != j :
            w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapfg[j],purify_e=False, purify_b=True), b)
        if i==j :
            w_dc.compute_coupling_matrix(nmt.NmtField(mask, 1*mapfg[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*mapfg[j],purify_e=False, purify_b=True), b)
        wsp_dc.append(w_dc)
 
wsp_dc=np.array(wsp_dc)

CLcross=np.zeros((1,Ncross,len(leff)))

#addition du bruit aux cartes
maptotaldc= mapfg  
z=0
for i in range(0,N_freqs):
	for j in range(i,N_freqs):
		CLcross[0,z]=np.array((sim.compute_master(nmt.NmtField(mask, 1*maptotaldc[i],purify_e=False, purify_b=True), nmt.NmtField(mask, 1*maptotaldc[j],purify_e=False, purify_b=True), wsp_dc[z]))[3])
		z = z +1  
np.save("./CLsimus/DLcross_fg_nside%s_fsky%s_scale%s_Nlbin%s_d%s"%(nside,fsky,scale,Nlbin,dusttype),leff*(leff+1)*CLcross/2/np.pi)

DLdust= leff*(leff+1)*CLcross/2/np.pi

nucross = []
for i in range(0,nf):
    for j in range(i,nf):
        nucross.append(np.sqrt(freq[i]*freq[j]))
nucross = np.array(nucross)

N = len(DLdc[:,0,0]) #in order to have a quicker run, replace by e.g. 50 or 100 here for testing.

DLdc=DLdc[:N,:,:Nell]

#compute Cholesky matrix:

Linvdc=an.getLinvdiag(DLdc)

p0=[5e2, 1.54, 20, 0] #first guess for mbb A, beta, T, r

resultsmbb = an.fitmbb(nucross,DLdust,Linvdc,p0)

dust=sky.components[0]
Adust= sim.downgrade_map(dust.get_emission(353 * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(353*u.GHz)),nside_in=512,nside_out=nside)
plt.plot(freq,mapfg[:,0,0])
plt.plot(freq,Adust[1,0]*func.MBB_fit(freq,1.54,20)/func.MBB_fit(353,1.54,20))
plt.show()

plt.plot(nucross,DLdust[0,0])

