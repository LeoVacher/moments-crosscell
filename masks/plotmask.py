import pymaster as nmt 
import pysm
from pysm.nominal import models
import time


nside = 256
Npix = hp.nside2npix(nside)
N=500 
Nf=9
lmax = nside*3-1
#lmax=850
scale = 5
Nlbin = 10
fsky = 0.6
dusttype = 1
Ncross = 45

# Initialize binning scheme with Nlbin ells per bandpower
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()

mask0 = hp.read_map("/Users/StarTraveller2.0/Documents/ProjetsRecherche/Moments/masks/mask_nside512_fsky%spc_P353_smooth10deg_cut53muK_nohole.fits"%(np.int(0.5*100)))
mask0 = hp.ud_grade(mask0,nside_out=nside)
mask1 = mask0
#mask1 = nmt.mask_apodization(mask0, scale, apotype='C2')

mask0 = hp.read_map("/Users/StarTraveller2.0/Documents/ProjetsRecherche/Moments/masks/mask_nside512_fsky%spc_P353_smooth10deg_cut80muK.fits"%(np.int(0.6*100)))
mask0 = hp.ud_grade(mask0,nside_out=nside)
#mask2 = nmt.mask_apodization(mask0, scale, apotype='C2')
mask2=mask0

mask0 = hp.read_map("/Users/StarTraveller2.0/Documents/ProjetsRecherche/Moments/masks/mask_nside512_fsky%spc_P353_smooth10deg_cut121muK.fits"%(np.int(0.7*100)))
mask0 = hp.ud_grade(mask0,nside_out=nside)
#mask3 = nmt.mask_apodization(mask0, scale, apotype='C2')
mask3=mask0

hp.mollview(mask1+mask2+mask3,cbar=False,title='')
plt.savefig(Pr+"Moments/FITPlots/article/masks.pdf")
plt.show()