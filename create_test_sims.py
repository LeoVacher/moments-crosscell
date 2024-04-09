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
nside = 512
Npix = hp.nside2npix(nside)
N=1000 
lmax = nside*3-1
#lmax=850
scale = 5
Nlbin = 10
fsky = 0.7
dusttype = 0
syncrotype = 0
mascut=0
kw = ''

# instr param

instr_name='LiteBIRD_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
sensP= instr['sens_P']
b = nmt.bins.NmtBin(nside=nside,lmax=lmax,nlb=Nlbin)
leff = b.get_effective_ells()

#call foreground sky

sky = pysm3.Sky(nside=nside, preset_strings=['d%s'%dusttype,'s%s'%syncrotype])#,'s%s'%synctype])
#freq_maps = get_observation(instrument, sky, unit='uK_CMB')
mapfg= np.array([sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)) for f in range(len(freq))])

# call cmb

